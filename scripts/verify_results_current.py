#!/usr/bin/env python
"""
scripts/verify_results_current.py -- results must come from the code that is in the repo NOW.

Reads the git commit recorded in results.json and fails if any code directory changed between that commit
and HEAD (or the working tree).  Workflow: commit code -> run experiments -> seal -> commit results/.
Editing the evaluation code afterwards invalidates the results until they are regenerated.

    python scripts/verify_results_current.py results/latest/results.json [--dirs scripts prajna_core experiments evaluation configs]
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict, List, Optional

DEFAULT_DIRS = ["scripts", "prajna_core", "experiments", "evaluation", "configs"]


def _git(args: List[str], cwd: str) -> Optional[str]:
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def recorded_commit(doc: Dict[str, Any]) -> Optional[str]:
    for path in (("metadata", "git", "git_hash"), ("provenance", "git", "commit")):
        cur: Any = doc
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur:
            return str(cur)
    return None


def check(results_path: str, dirs: List[str], repo: str = ".") -> List[str]:
    with open(results_path, encoding="utf-8") as f:
        doc = json.load(f)
    commit = recorded_commit(doc)
    if not commit:
        return ["results.json does not record a git commit"]
    if _git(["cat-file", "-e", f"{commit}^{{commit}}"], repo) is None:
        return [f"recorded commit {commit[:8]} is not in this repository"]
    present = [d for d in dirs if (_git(["ls-tree", "-d", "--name-only", "HEAD", d], repo) or "").strip()]
    problems: List[str] = []
    changed = (_git(["diff", "--name-only", commit, "HEAD", "--"] + present, repo) or "").split()
    for f in changed:
        problems.append(f"changed since results were produced (commit {commit[:8]}): {f}")
    dirty = (_git(["status", "--porcelain", "--"] + present, repo) or "").splitlines()
    for line in dirty:
        problems.append(f"uncommitted change in code directory: {line.strip()}")
    return problems


def main(argv: List[str] = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        print(__doc__)
        return 2
    dirs = DEFAULT_DIRS
    if "--dirs" in a:
        i = a.index("--dirs")
        dirs, a = a[i + 1:], a[:i]
    problems = check(a[0], dirs)
    for p in problems:
        print(f"[STALE] {p}")
    print("[OK] results are current" if not problems else f"{len(problems)} problem(s): regenerate results on a clean tree")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
