#!/usr/bin/env python
"""
scripts/detect_suspicious_results.py -- runtime check for fabricated-looking numbers in results.json.

The static linter can miss a shortcut (a name that does not look like a metric, a value built in another file).
This tool looks at the RESULTS instead: independently trained models must not be related by an exact formula.

Checks on every per-seed / per-run array (>= 3 numbers) found anywhere in the JSON:
  C1 zero variance   identical values across seeds (a learned model with std 0.00 is almost never real)
  C2 constant ratio  a_i / b_i is the same for every seed (e.g. res_b = res_a * 0.42, model_c = model_b * 0.35)
  C3 exact linearity |corr(a, b)| > 0.99999 between two arrays (a = k*b + c)
Deterministic methods (rule-based, closed-form) may declare  "deterministic": true  in their dict to skip C1.

    python scripts/detect_suspicious_results.py results/latest/results.json
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from typing import Any, Dict, List, Tuple

ARRAY_KEYS = ("per_seed", "per_run", "values", "seed_values")


def _is_num_list(x: Any) -> bool:
    return (isinstance(x, list) and len(x) >= 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in x))


def collect_arrays(node: Any, path: str = "", deterministic: bool = False) -> List[Tuple[str, List[float], bool]]:
    out: List[Tuple[str, List[float], bool]] = []
    if isinstance(node, dict):
        det = deterministic or bool(node.get("deterministic"))
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if _is_num_list(v) and (any(s in str(k).lower() for s in ARRAY_KEYS)):
                out.append((p, [float(x) for x in v], det))
            else:
                out += collect_arrays(v, p, det)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += collect_arrays(v, f"{path}[{i}]", deterministic)
    return out


def _mean(v: List[float]) -> float:
    return sum(v) / len(v)


def _std(v: List[float]) -> float:
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _corr(a: List[float], b: List[float]) -> float:
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 0 else float("nan")


def analyse(doc: Dict[str, Any], max_arrays: int = 600) -> List[str]:
    arrays = collect_arrays(doc)[:max_arrays]
    flags: List[str] = []
    for path, v, det in arrays:
        if not det and _std(v) == 0.0:
            flags.append(f"C1 zero variance across {len(v)} seeds at {path} (value {v[0]:g}); "
                         f"set 'deterministic': true only for rule-based/closed-form methods")
    for (pa, a, _), (pb, b, _) in itertools.combinations(arrays, 2):
        if len(a) != len(b) or a == b or _std(a) == 0.0 or _std(b) == 0.0:
            continue
        ratios = [x / y for x, y in zip(a, b) if y != 0.0]
        if len(ratios) == len(a):
            m = _mean(ratios)
            if abs(m - 1.0) > 1e-3 and _std(ratios) / abs(m) < 1e-3:
                flags.append(f"C2 constant ratio {m:.4g} between {pa} and {pb} across all seeds "
                             f"(independent runs cannot be related by a fixed multiplier)")
                continue
        c = _corr(a, b)
        if not math.isnan(c) and abs(c) > 0.99999:
            flags.append(f"C3 exact linear relation (|corr|={abs(c):.7f}) between {pa} and {pb}")
    return flags


def main(argv: List[str] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__)
        return 2
    doc = json.load(open(args[0], encoding="utf-8"))
    n = len(collect_arrays(doc))
    flags = analyse(doc)
    for f in flags:
        print(f)
    print(f"{len(flags)} suspicious pattern(s) across {n} per-seed array(s) in {args[0]}")
    return 1 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
