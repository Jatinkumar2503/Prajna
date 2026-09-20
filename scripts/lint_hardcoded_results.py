#!/usr/bin/env python3
"""
PRAJNA Scientific Integrity Linter
==================================
Scans source code for:
  - Formula shortcuts (e.g. res_b = res_a * 0.42, acc = 1.0 - 0.0025 * h)
  - Hardcoded metric assignments or mock dictionaries
  - Synthetic random sampling around fixed metric targets (e.g. random.uniform(0.99, 1.0))
  - Typed-in metric(value=...) calls

Suppression:
  Add an inline comment: # provenance: allow (reason)
"""

import os
import sys
import re
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

IGNORED_DIRS = {".git", "__pycache__", ".idea", ".vscode", "node_modules", ".gemini", "venv", ".env"}

# Patterns that indicate formula shortcuts or synthetic result fabrication
SHORTCUT_PATTERNS = [
    (re.compile(r"(\bres_[a-z0-9_]+\s*=\s*res_[a-z0-9_]+\s*\*\s*0\.\d+)"), "Formula shortcut on residual"),
    (re.compile(r"(\*\s*0\.42\b)"), "Hardcoded 0.42 scaling shortcut"),
    (re.compile(r"(\*\s*0\.35\b)"), "Hardcoded 0.35 scaling shortcut"),
    (re.compile(r"(\bacc\s*=\s*1\.0\s*-\s*0\.\d+\s*\*)"), "Synthetic linear accuracy decay formula"),
    (re.compile(r"(\bnp\.random\.normal\s*\(\s*0\.99)"), "Synthetic accuracy generation around 0.99"),
    (re.compile(r"(\brandom\.uniform\s*\(\s*0\.99)"), "Synthetic accuracy generation around 0.99"),
    (re.compile(r"(\bmetric\s*\(\s*value\s*=\s*0\.\d+)"), "Typed-in metric(value=...) call"),
]

def lint_file(filepath):
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, start=1):
                raw = line.strip()
                # Skip comments and empty lines
                if not raw or raw.startswith("#"):
                    continue
                # Check for suppression
                if "provenance: allow" in line:
                    continue

                for pattern, desc in SHORTCUT_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        findings.append((line_idx, desc, match.group(1), raw))
    except Exception as e:
        pass
    return findings

def main():
    target_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["scripts", "experiments", "prajna_core", "evaluation"]
    print(f"[*] Linting directories for hardcoded shortcuts & synthetic results: {target_dirs}")

    total_findings = 0
    self_path = Path(__file__).resolve()

    for d in target_dirs:
        d_path = WORKSPACE_ROOT / d if not Path(d).is_absolute() else Path(d)
        if not d_path.exists():
            continue

        for root, dirs, files in os.walk(d_path):
            dirs[:] = [sub for sub in dirs if sub not in IGNORED_DIRS]
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = Path(root) / file
                if file_path.resolve() == self_path:
                    continue

                findings = lint_file(file_path)
                if findings:
                    rel = file_path.relative_to(WORKSPACE_ROOT) if file_path.is_relative_to(WORKSPACE_ROOT) else file_path
                    for line_idx, desc, match_str, content in findings:
                        print(f"[LINT FINDING] {rel}:{line_idx}: {desc} ('{match_str}')\n  Line: {content}")
                        total_findings += 1

    if total_findings > 0:
        print(f"\n[FAIL] Found {total_findings} lint findings. Fix calculations or add '# provenance: allow (reason)'.")
        sys.exit(1)
    else:
        print("[+] Zero hardcoded result shortcuts detected across target directories.")
        sys.exit(0)

if __name__ == "__main__":
    main()
