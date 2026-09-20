#!/usr/bin/env python3
"""
PRAJNA Retracted Numbers Scanner
================================
Greps the codebase for numbers that originated from previous calculation bugs,
legacy formula shortcuts, or conflicting iterations.

Usage:
  python scripts/find_retracted_numbers.py .
"""

import os
import sys
import re
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

IGNORED_DIRS = {".git", "__pycache__", ".idea", ".vscode", "node_modules", ".gemini", "venv", ".env", "checkpoints", "artifacts", "raw", "data", "harmonized"}
ALLOWED_EXTS = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".js", ".html", ".css", ".bat", ".ps1"}

def load_retracted_numbers():
    list_path = Path(__file__).parent / "retracted_numbers.txt"
    if not list_path.exists():
        print(f"[!] {list_path} not found.")
        return []
    numbers = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                numbers.append(line)
    return numbers

def scan_file(filepath, retracted_numbers):
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, start=1):
                # Check for suppression
                if "retracted-ok:" in line:
                    continue
                for num in retracted_numbers:
                    # Match number as a distinct numeric token (not as a prefix of another float)
                    pattern = re.compile(r'(?<![0-9.])' + re.escape(num) + r'(?![0-9.])')
                    if pattern.search(line):
                        findings.append((line_idx, num, line.strip()))
    except Exception as e:
        pass
    return findings

def main():
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else WORKSPACE_ROOT
    retracted_numbers = load_retracted_numbers()

    if not retracted_numbers:
        print("[*] No retracted numbers loaded. Exiting clean.")
        sys.exit(0)

    print(f"[*] Scanning {target_dir} for {len(retracted_numbers)} retracted numbers...")
    total_findings = 0

    self_path = Path(__file__).resolve()
    list_path = (Path(__file__).parent / "retracted_numbers.txt").resolve()

    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for file in files:
            file_path = Path(root) / file
            if file_path.resolve() in (self_path, list_path):
                continue
            if file_path.suffix.lower() not in ALLOWED_EXTS:
                continue

            findings = scan_file(file_path, retracted_numbers)
            if findings:
                rel_path = file_path.relative_to(WORKSPACE_ROOT) if file_path.is_relative_to(WORKSPACE_ROOT) else file_path
                for line_idx, num, content in findings:
                    print(f"[RETRACTED NUMBER HIT] {rel_path}:{line_idx} contains '{num}': {content}")
                    total_findings += 1

    if total_findings > 0:
        print(f"\n[FAIL] Found {total_findings} instances of retracted numbers. Fix or add 'retracted-ok: (reason)'.")
        sys.exit(1)
    else:
        print("[+] Zero retracted numbers detected across codebase.")
        sys.exit(0)

if __name__ == "__main__":
    main()
