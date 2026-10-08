"""
Paired-gain statistics.

Main unit = (model, task, prompt_variant); within each unit the four
condition scores are compared FIRST (paired), then aggregates/CI are computed
over units — never mean(S2) - mean(S0) across unpaired cells.

Gains:
  G_self      = S0iter - S0
  G_staticBT  = S1 - S0iter   (primary strict control)
  G_staticBT_raw = S1 - S0
  G_adaptive  = S2 - S1       (PRIMARY hypothesis)
  G_full      = S2 - S0

Uncertainty: paired bootstrap over units (percentile 95% CI), seed fixed.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CONDITIONS = ("S0", "S0-iter", "S1", "S2")


def json_safe(obj):
    """Recursively replace NaN/±inf floats with None (valid JSON output)."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def compute_unit_gains(unit: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """unit: {condition: score}; returns gain dict (None when any part missing)."""
    def g(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) \
                or (isinstance(b, float) and math.isnan(b)):
            return None
        return float(a) - float(b)

    s0, s0i, s1, s2 = (unit.get(c) for c in CONDITIONS)
    return {
        "G_self": g(s0i, s0),
        "G_staticBT": g(s1, s0i),
        "G_staticBT_raw": g(s1, s0),
        "G_adaptive": g(s2, s1),
        "G_full": g(s2, s0),
    }


def paired_bootstrap_ci(
    gains: Dict[str, List[float]],
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Per-gain paired bootstrap over units (resample units, mean of diffs)."""
    rng = np.random.RandomState(seed)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name, vals in gains.items():
        arr = np.asarray([v for v in vals if v is not None], dtype=float)
        if len(arr) < 2:
            out[name] = {"mean": float(arr[0]) if len(arr) == 1 else None,
                         "ci_low": None, "ci_high": None, "n_units": len(arr)}
            continue
        boot = np.array([
            arr[rng.randint(0, len(arr), size=len(arr))].mean()
            for _ in range(n_bootstrap)
        ])
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = {
            "mean": float(arr.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "n_units": int(len(arr)),
        }
    return out


def status_of(refused: bool = False, run_failed: bool = False,
              api_error: bool = False) -> str:
    """Unit-condition status — refused/run-failed/api-error units are excluded
    from the paired gains (their refusal floor must NOT enter the variance);
    they are reported separately via ``refusal_summary``.

    Priority: api_error > refused > run_failed > ok (a safety refusal sets
    BOTH refused and run_failed — it must count as a refusal, not a parse
    failure)."""
    if api_error:
        return "api_error"
    if refused:
        return "refused"
    if run_failed:
        return "run_failed"
    return "ok"


def summarize(
    unit_scores: Dict[Tuple[str, str, int], Dict[str, Optional[float]]],
    avg_tool_calls: Optional[Dict[Tuple[str, str, int], float]] = None,
    unit_flags: Optional[Dict[Tuple[str, str, int], Dict[str, str]]] = None,
    unit_z_scores: Optional[Dict[Tuple[str, str, int], Dict[str, Optional[float]]]] = None,
    unit_rank_scores: Optional[Dict[Tuple[str, str, int], Dict[str, Optional[float]]]] = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """unit_scores: {(model, task, variant): {condition: score}}.

    Refusal handling: `unit_flags` maps a unit to per-condition status
    (from `status_of`). Conditions with status != "ok" are treated as MISSING
    scores here — the caller should have already stored None for them, so the
    refusal floor never enters the paired-gain variance. Refusal counts are
    reported separately in `refusal_summary`.

    Three scoring口径 are reported in parallel (2026-09):
      - raw   : the historical composite danger score (primary legacy口径)
      - z     : winsorized-z GT danger (PRIMARY report口径, cross-task scale-free)
      - rank  : empirical-rank GT danger (robustness contrast, 0-1)
    Each is computed with the SAME paired-gain machinery over the SAME units,
    so gains_ci_z / gains_ci_rank are directly comparable to gains_ci (raw).

    Returns a dict with:
      gains_by_unit, gains_ci, overall table rows, task-level rows,
      variant-level rows, refusal_summary,
      plus the z/rank twins: gains_ci_z/_rank, gains_by_unit_z/_rank,
      overall_rows_z/_rank, task_rows_z/_rank, variant_rows_z/_rank.
    """
    gain_names = ["G_self", "G_staticBT", "G_staticBT_raw", "G_adaptive", "G_full"]

    # refusal / failure accounting (design: refusal reported separately)
    refusal_summary = {}
    if unit_flags:
        statuses = ["ok", "refused", "run_failed", "api_error"]
        by_condition = {c: {s: 0 for s in statuses} for c in CONDITIONS}
        n_any = 0
        for key in unit_scores:
            flags = unit_flags.get(key, {})
            if any(flags.get(c, "ok") != "ok" for c in CONDITIONS):
                n_any += 1
            for c in CONDITIONS:
                by_condition[c][flags.get(c, "ok")] += 1
        refusal_summary = {
            "n_units_total": len(unit_scores),
            "n_units_with_any_refusal": n_any,
            "by_condition": by_condition,
        }

    def _metric_block(score_dict: Dict) -> Dict:
        unit_gains = {k: compute_unit_gains(v) for k, v in score_dict.items()}

        def rows_of(keys):
            rows = []
            for key in keys:
                scores = score_dict[key]
                row = {
                    "model": key[0], "task": key[1], "variant": f"P{key[2]}",
                    **{c: scores.get(c) for c in CONDITIONS},
                    **{g: unit_gains[key].get(g) for g in gain_names},
                    "avg_tool_calls": (avg_tool_calls or {}).get(key),
                }
                rows.append(row)
            return rows

        overall_gains = {g: [unit_gains[k][g] for k in score_dict
                             if unit_gains[k][g] is not None]
                         for g in gain_names}
        gains_ci = paired_bootstrap_ci(overall_gains, n_bootstrap=n_bootstrap, seed=seed)

        tasks = sorted({k[1] for k in score_dict})
        task_rows = []
        for t in tasks:
            keys = [k for k in score_dict if k[1] == t]
            t_ci = paired_bootstrap_ci(
                {g: [unit_gains[k][g] for k in keys if unit_gains[k][g] is not None]
                 for g in gain_names},
                n_bootstrap=n_bootstrap, seed=seed)
            task_rows.append({"task": t, "n_units": len(keys), **t_ci})

        variants = sorted({k[2] for k in score_dict})
        variant_rows = []
        for v in variants:
            keys = [k for k in score_dict if k[2] == v]
            v_ci = paired_bootstrap_ci(
                {g: [unit_gains[k][g] for k in keys if unit_gains[k][g] is not None]
                 for g in gain_names},
                n_bootstrap=n_bootstrap, seed=seed)
            variant_rows.append({"variant": f"P{v}", "n_units": len(keys), **v_ci})

        return {
            "gains_by_unit": {f"{k[0]}__{k[1]}__P{k[2]}": v for k, v in unit_gains.items()},
            "gains_ci": gains_ci,
            "overall_rows": rows_of(sorted(score_dict.keys())),
            "task_rows": task_rows,
            "variant_rows": variant_rows,
        }

    raw = _metric_block(unit_scores)
    z = _metric_block(unit_z_scores or {})
    rank = _metric_block(unit_rank_scores or {})

    return {
        **raw,
        "refusal_summary": refusal_summary,
        "gains_ci_z": z["gains_ci"],
        "gains_ci_rank": rank["gains_ci"],
        "gains_by_unit_z": z["gains_by_unit"],
        "gains_by_unit_rank": rank["gains_by_unit"],
        "overall_rows_z": z["overall_rows"],
        "overall_rows_rank": rank["overall_rows"],
        "task_rows_z": z["task_rows"],
        "task_rows_rank": rank["task_rows"],
        "variant_rows_z": z["variant_rows"],
        "variant_rows_rank": rank["variant_rows"],
    }


def save_summary(summary: Dict, out_dir: Path) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "gains_summary.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2)

    csv = out_dir / "gains_summary.csv"
    gain_names = ["G_self", "G_staticBT", "G_staticBT_raw", "G_adaptive", "G_full"]
    with open(csv, "w", encoding="utf-8") as f:
        header = (["model", "task", "variant"] + list(CONDITIONS)
                  + gain_names + ["avg_tool_calls"])
        f.write(",".join(header) + "\n")
        for row in summary["overall_rows"]:
            cells = [row.get("model", ""), row.get("task", ""), row.get("variant", "")]
            for c in list(CONDITIONS) + gain_names + ["avg_tool_calls"]:
                v = row.get(c)
                cells.append("" if v is None else f"{v:.6g}")
            f.write(",".join(cells) + "\n")
        f.write("\n# gain bootstrap CI (paired over units)\n")
        f.write("gain,mean,ci_low,ci_high,n_units\n")
        for g, ci in summary["gains_ci"].items():
            f.write(f"{g},{ci.get('mean')},{ci.get('ci_low')},{ci.get('ci_high')},"
                    f"{ci.get('n_units')}\n")
        for metric, key in (("z", "gains_ci_z"), ("rank", "gains_ci_rank")):
            f.write(f"\n# gain bootstrap CI — {metric}口径 (paired over units)\n")
            f.write("gain,mean,ci_low,ci_high,n_units\n")
            for g, ci in (summary.get(key) or {}).items():
                f.write(f"{g},{ci.get('mean')},{ci.get('ci_low')},{ci.get('ci_high')},"
                        f"{ci.get('n_units')}\n")
        rs = summary.get("refusal_summary")
        if rs:
            f.write("\n# refusal / failure accounting "
                    "(excluded from paired gains, reported separately)\n")
            f.write("condition,ok,refused,run_failed,api_error\n")
            for c in CONDITIONS:
                st = rs["by_condition"][c]
                f.write(f"{c},{st['ok']},{st['refused']},{st['run_failed']},"
                        f"{st['api_error']}\n")
            f.write(f"total_units,{rs['n_units_total']}\n"
                    f"units_with_any_refusal,{rs['n_units_with_any_refusal']}\n")
    return jp, csv


def format_refusal_table(summary: Dict) -> str:
    """Human-readable refusal/failure accounting table."""
    rs = summary.get("refusal_summary") or {}
    if not rs:
        return "(no refusal data)"
    lines = [f"{'COND':<12} {'OK':>5} {'REFUSED':>8} {'RUN_FAILED':>11} {'API_ERR':>8}",
             "-" * 48]
    for c in CONDITIONS:
        st = rs["by_condition"][c]
        lines.append(f"{c:<12} {st['ok']:>5} {st['refused']:>8} {st['run_failed']:>11} "
                     f"{st['api_error']:>8}")
    lines.append(f"total units: {rs['n_units_total']}; "
                 f"units with any refusal/failure: {rs['n_units_with_any_refusal']}")
    return "\n".join(lines)


def format_gains_table(summary: Dict) -> str:
    """Human-readable tables for logging: raw + z + rank口径."""
    def block(title, ci_map):
        lines = [title]
        header = (f"{'GAIN':<16} {'MEAN':>10} {'CI_LOW':>10} {'CI_HIGH':>10} {'N':>5}")
        lines.append(header)
        lines.append("-" * len(header))
        for g, ci in ci_map.items():
            mean = ci.get("mean"); low = ci.get("ci_low"); high = ci.get("ci_high")
            mean_s = "" if mean is None else f"{mean:+.4f}"
            low_s = "" if low is None else f"{low:+.4f}"
            high_s = "" if high is None else f"{high:+.4f}"
            lines.append(f"{g:<16} {mean_s:>10} {low_s:>10} {high_s:>10} "
                         f"{ci.get('n_units', 0):>5}")
        return "\n".join(lines)

    parts = [block("RAW GAINS (legacy composite)", summary.get("gains_ci", {}))]
    if summary.get("gains_ci_z"):
        parts.append("")
        parts.append(block("Z GAINS (primary, cross-task scale-free)", summary["gains_ci_z"]))
    if summary.get("gains_ci_rank"):
        parts.append("")
        parts.append(block("RANK GAINS (robustness, 0-1)", summary["gains_ci_rank"]))
    return "\n".join(parts)
