#!/usr/bin/env python3
"""
PRAJNA Results Cryptographic Sealer & Verifier
==============================================
Seals results.json by computing the canonical SHA-256 hash across all fields,
rendering the document tamper-evident.

Usage:
  python scripts/seal_results.py seal results/latest/results.json [--require-clean]
  python scripts/seal_results.py verify results/latest/results.json [--require-clean]
"""

import sys
import json
import hashlib
import datetime
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def check_git_clean():
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=WORKSPACE_ROOT
        ).decode("utf-8").strip()
        return len(output) == 0, output
    except Exception as e:
        return False, str(e)

def compute_canonical_hash(doc):
    # Shallow copy and remove seal from metadata if present
    doc_copy = json.loads(json.dumps(doc))
    if "metadata" in doc_copy and "seal" in doc_copy["metadata"]:
        del doc_copy["metadata"]["seal"]

    canonical_str = json.dumps(doc_copy, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

def seal_file(filepath, require_clean=False):
    target = Path(filepath)
    if not target.exists():
        print(f"[FAIL] Target file does not exist: {target}")
        sys.exit(1)

    if require_clean:
        is_clean, dirty_status = check_git_clean()
        if not is_clean:
            print(f"[FAIL] --require-clean specified but working tree has uncommitted changes:\n{dirty_status}")
            sys.exit(1)

    with open(target, "r", encoding="utf-8") as f:
        doc = json.load(f)

    if "metadata" not in doc:
        doc["metadata"] = {}

    seal_hash = compute_canonical_hash(doc)
    doc["metadata"]["seal"] = {
        "algorithm": "sha256",
        "canonical_sha256": seal_hash,
        "sealed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    with open(target, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    sig_file = target.with_suffix(".json.sha256")
    with open(sig_file, "w", encoding="utf-8") as f:
        f.write(f"{seal_hash}  {target.name}\n")

    print(f"[+] Sealed {target} with canonical SHA-256:\n    {seal_hash}")
    sys.exit(0)

def verify_file(filepath, require_clean=False):
    target = Path(filepath)
    if not target.exists():
        print(f"[FAIL] Target file does not exist: {target}")
        sys.exit(1)

    with open(target, "r", encoding="utf-8") as f:
        doc = json.load(f)

    if require_clean:
        is_clean, dirty_status = check_git_clean()
        if not is_clean:
            print(f"[FAIL] --require-clean specified but working tree has uncommitted changes:\n{dirty_status}")
            sys.exit(1)

        is_dirty_meta = doc.get("metadata", {}).get("git", {}).get("is_dirty", False)
        if is_dirty_meta:
            print("[FAIL] --require-clean specified but results.json metadata records is_dirty: true")
            sys.exit(1)

    seal_info = doc.get("metadata", {}).get("seal", {})
    recorded_hash = seal_info.get("canonical_sha256")
    if not recorded_hash:
        print(f"[FAIL] No canonical seal found in metadata of {target}")
        sys.exit(1)

    computed_hash = compute_canonical_hash(doc)
    if computed_hash != recorded_hash:
        print(f"[FAIL] Canonical SHA-256 seal mismatch in {target}!")
        print(f"  Recorded: {recorded_hash}")
        print(f"  Computed: {computed_hash}")
        sys.exit(1)

    print(f"[SUCCESS] Cryptographic seal verified for {target.name}:\n    {computed_hash}")
    sys.exit(0)

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scripts/seal_results.py seal <path_to_results.json> [--require-clean]")
        print("  python scripts/seal_results.py verify <path_to_results.json> [--require-clean]")
        sys.exit(1)

    action = sys.argv[1].lower()
    filepath = sys.argv[2]
    require_clean = "--require-clean" in sys.argv

    if action == "seal":
        seal_file(filepath, require_clean)
    elif action == "verify":
        verify_file(filepath, require_clean)
    else:
        print(f"[!] Unknown action '{action}'. Use 'seal' or 'verify'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
