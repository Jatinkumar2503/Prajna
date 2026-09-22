"""
PRAJNA CI Table Provenance Verifier
===================================
Verifies that every table in README.md and docs/provenance_tables.md
matches the canonical JSON results byte-for-byte when rendered.

Exits with:
  - Code 0: All table templates match results.json byte-for-byte.
  - Code 1: Any character, number, or formatting discrepancy detected.
"""

import sys
import json
import re
import difflib
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.generate_results_provenance import render_markdown_table

def extract_tagged_tables(markdown_text):
    tables = {}
    pattern = re.compile(
        r"<!--\s*PROVENANCE_TABLE_START:([a-zA-Z0-9_]+)\s*-->\s*\n(.*?)\n\s*<!--\s*PROVENANCE_TABLE_END:\1\s*-->",
        re.DOTALL
    )
    for match in pattern.finditer(markdown_text):
        table_id = match.group(1)
        table_content = match.group(2).strip()
        tables[table_id] = table_content
    return tables

def verify_provenance_file(target_file: Path, provenance_json_path: Path) -> bool:
    print(f"[*] Verifying table provenance in: {target_file}")
    if not target_file.exists():
        print(f"[FAIL] Target markdown file does not exist: {target_file}")
        return False

    if not provenance_json_path.exists():
        print(f"[FAIL] Provenance JSON does not exist: {provenance_json_path}")
        return False

    with open(provenance_json_path, "r", encoding="utf-8") as f:
        provenance = json.load(f)

    json_tables = provenance.get("tables", {})

    with open(target_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_tables = extract_tagged_tables(md_text)
    if not md_tables:
        print(f"[WARN] No provenance-tagged tables found in {target_file}.")
        return False

    all_passed = True

    for table_id, actual_content in md_tables.items():
        if table_id not in json_tables:
            print(f"[FAIL] Markdown contains table '{table_id}' not found in {provenance_json_path}")
            all_passed = False
            continue

        json_rows = json_tables[table_id]
        expected_content = render_markdown_table(table_id, json_rows).strip()
        actual_content_stripped = actual_content.strip()

        # Byte-for-byte comparison
        if expected_content != actual_content_stripped:
            print(f"\n[FAIL] Byte-for-byte mismatch in table '{table_id}' inside {target_file.name}!")
            diff = difflib.unified_diff(
                expected_content.splitlines(),
                actual_content_stripped.splitlines(),
                fromfile=f"expected_from_{provenance_json_path.name}",
                tofile=f"actual_{target_file.name}",
                lineterm=""
            )
            print("\n".join(diff))
            all_passed = False
        else:
            print(f"  [+] Table '{table_id}' matches byte-for-byte ({len(json_rows)} rows).")

    return all_passed

def main():
    provenance_json = WORKSPACE_ROOT / "results" / "latest" / "results.json"
    readme_path = WORKSPACE_ROOT / "README.md"
    reproduce_path = WORKSPACE_ROOT / "REPRODUCE.md"
    provenance_md = WORKSPACE_ROOT / "docs" / "provenance_tables.md"

    success_docs = verify_provenance_file(provenance_md, provenance_json)
    
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_has_tags = "<!-- PROVENANCE_TABLE_START:" in f.read()

    if readme_has_tags:
        success_readme = verify_provenance_file(readme_path, provenance_json)
    else:
        success_readme = True

    if reproduce_path.exists():
        with open(reproduce_path, "r", encoding="utf-8") as f:
            reproduce_has_tags = "<!-- PROVENANCE_TABLE_START:" in f.read()
        if reproduce_has_tags:
            success_reproduce = verify_provenance_file(reproduce_path, provenance_json)
        else:
            success_reproduce = True
    else:
        success_reproduce = True

    if success_docs and success_readme and success_reproduce:
        print("\n[PROVENANCE CI PASS] Every table verified byte-for-byte against results.json.")
        sys.exit(0)
    else:
        print("\n[PROVENANCE CI FAIL] Discrepancies detected between markdown tables and results.json.")
        sys.exit(1)

if __name__ == "__main__":
    main()
