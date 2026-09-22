#!/usr/bin/env python
"""
scripts/lint_hardcoded_results.py -- Step 2: static check that metrics are COMPUTED, not typed in.

Rules
  R1 literal      accuracy = 0.97 ; results["mae"] = 0.33
  R2 formula      res_b = res_a * 0.42 ; acc = 1.0 - 0.0025 * h   (constants + names, no function call)
  R3 literal list accs = [0.99, 0.97, 0.95]
  R4 dict literal {"mae": 0.33, "coverage": 0.9}
  R5 record call writer.metric("x", value=0.33) / values=[literals]
  R6 synthetic    acc = rng.normal(0.95, 0.01)   (random numbers drawn around a chosen target)

Dataclass field defaults (class bodies) are treated as configuration and skipped.
Suppress a deliberate case with a trailing comment:   # provenance: allow (reason)

    python scripts/lint_hardcoded_results.py scripts experiments prajna_core evaluation
Exit code 1 if any finding. No third-party imports.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

METRIC_RE = re.compile(
    r"(acc|mae|rmse|mse|r2|residual|coverage|latency|recall|precision|f1|loss|lead|error|auc|"
    r"throughput|false_alarm|fpr|tpr|nuisance|violation|(^|_)res($|_|\d)|(^|_)err($|_|\d))", re.I)
CONFIG_RE = re.compile(
    r"(tol|thresh|limit|target|alpha|\blr\b|eps|seed|budget|epoch|batch|weight|lambda|nominal|bound|"
    r"window|step|scale|clip|warn|crit|default|init|max_|min_|_max|_min|expected|patience|decay|"
    r"ramp|time|delay|tau)", re.I)
RANDOM_FUNCS = {"normal", "uniform", "random", "randn", "rand", "choice", "integers", "randint", "gauss"}
ALLOW_MARK = "provenance: allow"


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


def _is_metric_name(name: Optional[str]) -> bool:
    return bool(name) and bool(METRIC_RE.search(name)) and not CONFIG_RE.search(name)


def _target_names(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return [sl.value]
        return _target_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        out: List[str] = []
        for e in node.elts:
            out += _target_names(e)
        return out
    return []


def _num(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _num(node.operand)
        return -v if (v is not None and isinstance(node.op, ast.USub)) else v
    return None


def _nonzero_literal(node: ast.AST) -> bool:
    v = _num(node)
    return v is not None and v != 0.0


def _has_call(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


def _has_fractional_constant(node: ast.AST) -> bool:
    for n in ast.walk(node):
        v = _num(n) if isinstance(n, (ast.Constant, ast.UnaryOp)) else None
        if v is not None and v != int(v):
            return True
    return False


def _numeric_list(node: ast.AST) -> bool:
    return (isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 2
            and all(_num(e) is not None for e in node.elts))


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: List[str]):
        self.path, self.lines, self.findings = path, lines, []
        self._class_body_nodes = set()

    def _add(self, node: ast.AST, rule: str, msg: str) -> None:
        line = node.lineno
        if ALLOW_MARK in (self.lines[line - 1] if 0 < line <= len(self.lines) else ""):
            return
        self.findings.append(Finding(self.path, line, rule, msg))

    def _check_value(self, node: ast.AST, names: List[str], value: ast.AST) -> None:
        metric_names = [n for n in names if _is_metric_name(n)]
        if not metric_names:
            return
        n = metric_names[0]
        if _nonzero_literal(value):
            self._add(node, "R1", f"metric '{n}' assigned a literal number; compute it from data")
        elif _numeric_list(value):
            self._add(node, "R3", f"metric '{n}' assigned a list of literal numbers")
        elif isinstance(value, ast.BinOp) and not _has_call(value) and _has_fractional_constant(value):
            self._add(node, "R2", f"metric '{n}' derived by formula from constants/other metrics "
                                  f"(looks like a hardcoded shortcut such as res_b = res_a * 0.42)")
        elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
              and value.func.attr in RANDOM_FUNCS and any(_num(a) is not None for a in value.args)):
            self._add(node, "R6", f"metric '{n}' drawn from a random generator around chosen numbers")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for stmt in node.body:                       # dataclass fields / class constants = configuration
            if isinstance(stmt, (ast.AnnAssign, ast.Assign)):
                self._class_body_nodes.add(id(stmt))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if id(node) not in self._class_body_nodes:
            names: List[str] = []
            for t in node.targets:
                names += _target_names(t)
            self._check_value(node, names, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and id(node) not in self._class_body_nodes:
            self._check_value(node, _target_names(node.target), node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        names = _target_names(node.target)
        c = _num(node.value)
        if (isinstance(node.op, (ast.Mult, ast.Div)) and any(_is_metric_name(n) for n in names)
                and c is not None and c != int(c)):
            self._add(node, "R2", f"metric '{names[0]}' scaled in place by a constant")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and isinstance(k.value, str) and _is_metric_name(k.value)
                    and (_nonzero_literal(v) or _numeric_list(v))):
                self._add(node, "R4", f"dict literal gives metric '{k.value}' a literal value")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"metric", "proportion", "rate_upper_bound", "record"}:
            for kw in node.keywords:
                if kw.arg in {"value", "values", "k", "events"} and (_nonzero_literal(kw.value) or _numeric_list(kw.value)):
                    self._add(node, "R5", f"{node.func.attr}(... {kw.arg}=<literal>) records a typed-in number")
        self.generic_visit(node)


def lint_source(source: str, path: str = "<string>") -> List[Finding]:
    v = _Visitor(path, source.splitlines())
    v.visit(ast.parse(source))
    return v.findings


def lint_paths(paths: Iterable[str]) -> List[Finding]:
    findings: List[Finding] = []
    for p in paths:
        root = Path(p)
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            if any(part in {".git", "__pycache__", ".venv", "venv", "site-packages"} for part in f.parts):
                continue
            try:
                findings += lint_source(f.read_text(encoding="utf-8"), str(f))
            except SyntaxError as e:
                findings.append(Finding(str(f), e.lineno or 0, "SYNTAX", str(e)))
    return findings


def list_allows(paths: Iterable[str]) -> List[str]:
    """Every suppression in the tree, so a reviewer can audit the reasons."""
    out: List[str] = []
    for p in paths:
        root = Path(p)
        if not root.exists():
            continue
        for f in ([root] if root.is_file() else sorted(root.rglob("*.py"))):
            if "__pycache__" in f.parts:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if ALLOW_MARK in line and "ALLOW_MARK" not in line and '"provenance: allow"' not in line:
                    out.append(f"{f}:{i}: {line.strip()}")
    return out


def main(argv: List[str] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    max_allows = None
    if "--max-allows" in args:
        i = args.index("--max-allows")
        max_allows = int(args[i + 1])
        del args[i:i + 2]
    show_allows = "--list-allows" in args
    args = [a for a in args if a != "--list-allows"]
    paths = args or ["scripts", "experiments", "prajna_core", "evaluation"]
    findings = lint_paths(paths)
    for f in findings:
        print(f)
    allows = list_allows(paths)
    if show_allows:
        for a in allows:
            print("allow:", a)
    print(f"{len(findings)} finding(s), {len(allows)} suppression(s) in {', '.join(paths)}")
    if max_allows is not None and len(allows) > max_allows:
        print(f"too many '# provenance: allow' suppressions ({len(allows)} > {max_allows}); justify or remove")
        return 1
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
