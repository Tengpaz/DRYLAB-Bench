#!/usr/bin/env python3
"""
Main experiment orchestration script.

Usage:
    python run_experiment.py --config config.yaml --data-dir ./data
    python run_experiment.py --step conditions --models gpt-4o
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Suppress noisy third-party loggers
for _noisy in ["httpx", "httpcore", "openai", "urllib3", "requests"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)
os.environ.setdefault("OPENAI_LOG", "warn")

from drylab_bench.data_loader import DataLoader, compute_danger_score, is_mutation_dangerous
from drylab_bench.llm_client import (
    create_all_clients,
    BaseLLMClient,
    LLMTimeoutError,
    LLMAPIError,
    detect_implicit_refusal,
)
from drylab_bench.evaluator import Evaluator

# LLM + biological-tools iterative benchmark
from drylab_bench.agent import AgentRunner, ConditionRunResult
from drylab_bench.prompts import register_task_spec_overrides
from drylab_bench.tools import PrecomputedTools
from drylab_bench.gains import (
    summarize,
    save_summary,
    format_gains_table,
    format_refusal_table,
    status_of,
    json_safe,
)


_SCRIPT_ROOT = Path(__file__).resolve().parent
_SCRIPT_FILES = [_SCRIPT_ROOT / "run_experiment.py"] + sorted(
    (_SCRIPT_ROOT / "drylab_bench").glob("*.py"))


def compute_script_version() -> str:
    """Eval-script version tag: f"{git_short}-{hash8}" (recorded in every run manifest)."""
    import hashlib
    import subprocess
    commit = "nogit"
    try:
        out = subprocess.run(
            ["git", "-C", str(_SCRIPT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            commit = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    h = hashlib.sha256()
    for p in _SCRIPT_FILES:
        if p.is_file():
            h.update(p.read_bytes())
    return f"{commit}-{h.hexdigest()[:8]}"

# ---------------------------------------------------------------------------
# Logging — the CWD experiment.log is appended (NOT rotated): concurrent
# run_experiment.py processes would race on the rotate/rename and crash or
# clobber each other's file. The authoritative per-run log lives inside each
# run directory (attach_run_log); the CWD copy is a convenience aggregate.
# ---------------------------------------------------------------------------

_LOG_FILE = "experiment.log"

# ============================================================================
# Pretty logging: "HH:MM:SS [LVL] message" — the level tag is blank for INFO
# so routine lines stay clean; WARN/ERROR/CRITICAL stand out. The terminal
# handler adds ANSI colors, the log file stays plain UTF-8 text.
# ============================================================================

_LEVEL_TAGS = {
    "DEBUG": "DBG", "INFO": "", "WARNING": "WARN", "ERROR": "ERROR",
    "CRITICAL": "CRIT",
}
_LEVEL_COLORS = {
    "DEBUG": "\033[90m", "WARNING": "\033[33m", "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class _PrettyFormatter(logging.Formatter):
    def __init__(self, color: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        tag = _LEVEL_TAGS.get(record.levelname, "")
        msg = record.getMessage()
        line = f"{ts}  {tag:>5}" if tag else f"{ts}  {'':>5}"
        line = f"{line}  {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if self.color and record.levelname in _LEVEL_COLORS:
            line = (f"{_LEVEL_COLORS[record.levelname]}{line}{_RESET}")
        return line


def _setup_logging() -> logging.Logger:
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(_PrettyFormatter(color=bool(getattr(sys.stdout, "isatty", lambda: False)())))
    file_h = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_h.setFormatter(_PrettyFormatter(color=False))
    logging.basicConfig(level=logging.INFO, handlers=[stream, file_h])
    return logging.getLogger("experiment")


def log_banner(title: str, detail: str = "") -> None:
    """Phase-level banner: a bold divider around the phase title."""
    logger.info("")
    logger.info("═" * 70)
    if detail:
        logger.info(f"  {title}")
        logger.info(f"  {detail}")
    else:
        logger.info(f"  {title}")
    logger.info("═" * 70)


# ---------------------------------------------------------------------------
# Run-archived log: a permanent copy of the FULL experiment log (all modules,
# since the handler is attached to the ROOT logger) is written into the run's
# own directory — results/run_*/experiment.log — so every experiment keeps
# its log forever instead of the CWD experiment.log being rotated away.
# ---------------------------------------------------------------------------

_RUN_LOG_HANDLER: Optional[logging.FileHandler] = None


def attach_run_log(run_dir: Path) -> None:
    """Attach a permanent experiment.log copy inside the run directory.

    Attached to the ROOT logger so module-level loggers (llm_client,
    data_loader, evaluator) are captured too. Only ONE run handler is ever
    attached — creating a new runner (e.g. in tests) replaces the previous.
    """
    global _RUN_LOG_HANDLER
    if _RUN_LOG_HANDLER is not None:
        logging.getLogger().removeHandler(_RUN_LOG_HANDLER)
        _RUN_LOG_HANDLER.close()
        _RUN_LOG_HANDLER = None
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    handler.setFormatter(_PrettyFormatter(color=False))
    logging.getLogger().addHandler(handler)
    _RUN_LOG_HANDLER = handler


logger = _setup_logging()

# Per-thread token for compact log lines
_tls = threading.local()


def _worker_prefix(model: str = "", task: str = "") -> str:
    """Compact prefix for worker thread logs."""
    if model and task:
        return f"[{model[:14]:<14} {task}]"
    elif model:
        return f"[{model[:14]:<14}]"
    return ""


# ============================================================================
# Experiment runner
# ============================================================================

class ExperimentRunner:

    def __init__(self, config: dict, data_dir: Path, output_dir: Path,
                 max_workers: int = 8, only_models: Optional[set] = None,
                 only_scenarios: Optional[set] = None):
        self.config = config
        # Scenario filter (--scenarios): restrict config["scenarios"] BEFORE
        # any component (DataLoader / Evaluator / loops) reads it, so data
        # loading, baselines and the condition runs run on the filtered set
        # only. Results of skipped scenarios are preserved on disk and merged
        # back (append mode) — this enables single-model / single-scenario runs.
        if only_scenarios:
            self.config["scenarios"] = [
                s for s in self.config["scenarios"] if s["id"] in only_scenarios
            ]
        self.only_scenarios = only_scenarios
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Permanent copy of the full experiment log inside this run's dir
        attach_run_log(self.output_dir)
        self.max_workers = max_workers
        # append mode: only these model ids are evaluated this run (results
        # of other models are preserved on disk and merged at save time)
        self.only_models = only_models

        # API error accounting — models that keep failing (timeout / 5xx /
        # connection) are EXCLUDED from the experiment; these are infrastructure
        # errors, NOT refusals (see _register_api_error).
        self._api_errors: Dict[str, int] = {}
        self._excluded_models: set = set()
        self.max_api_errors = 3   # API errors tolerated per model before exclusion

        # This benchmark runs the LLM + biological-tools iterative conditions
        # pipeline (S0 / S0-iter / S1 / S2, with derived snapshots + warmup).
        self.unit_records: Dict[str, dict] = {}

        self.insilico = None            # InsilicoVerifier (built in _init_insilico)

        # Full-output logging (issue: preserve every experiment's raw data)
        self._output_lock = threading.Lock()
        self.outputs_path = self.output_dir / "outputs.jsonl"

        # Init
        log_banner("INITIALIZATION")
        logger.info("Run dir → %s", self.output_dir)
        logger.info("Loading data...")
        self.data_loader = DataLoader(data_dir, config)
        self.data_loader.load_all()

        logger.info("Creating clients...")
        self.clients = create_all_clients(config)

        # baseline_data must exist BEFORE the evaluator is built (the
        # evaluator holds a shared reference to it)
        self.baseline_data: Dict[str, Dict[str, np.ndarray]] = {}

        self.evaluator = Evaluator(self.data_loader, config, self.baseline_data)
        self._write_run_manifest()

    # ======================================================================
    # API-error accounting & model exclusion
    # ======================================================================

    def _register_api_error(self, m_id: str) -> None:
        """Count one API error for the model; exclude it past the threshold."""
        self._api_errors[m_id] = self._api_errors.get(m_id, 0) + 1
        n = self._api_errors[m_id]
        if n >= self.max_api_errors and m_id not in self._excluded_models:
            self._excluded_models.add(m_id)
            logger.error(
                "❌ EXCLUDED model %s — %d consecutive API errors (timeout/5xx/"
                "connection). Treating as ERROR, not refusal.",
                m_id, n,
            )

    def _filter_models(self) -> list:
        """Config models, optionally filtered to --models (append mode);
        API-error-excluded models are dropped."""
        if not self.only_models:
            models = list(self.config["models"])
        else:
            models = [m for m in self.config["models"] if m["id"] in self.only_models]
        return [m for m in models if m["id"] not in self._excluded_models]

    # ======================================================================
    # Full-output logging (every LLM call, preserved per experiment)
    # ======================================================================

    def _log_full_output(self, sc_id: str, m_id: str, approach: str, tag: str,
                         prompt: str, system: str, response_text: str,
                         temperature: float, refused: bool, api_error: bool,
                         refusal_reason: str, n_parsed: Optional[int] = None,
                         n_expected: Optional[int] = None,
                         extra: Optional[dict] = None) -> None:
        """Append one JSONL record per LLM call — the experiment's complete
        raw data (prompts + responses + classification). ``extra`` merges
        legacy fields (variant, schemes, sample_danger_score, ...)."""
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "scenario": sc_id, "model": m_id, "approach": approach, "tag": tag,
            "temperature": temperature, "refused": refused,
            "api_error": api_error, "refusal_reason": refusal_reason,
            "n_parsed": n_parsed, "n_expected": n_expected,
            "system_prompt": system, "prompt": prompt, "response": response_text,
        }
        if extra:
            rec.update(extra)
        with self._output_lock:
            with open(self.outputs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write_run_manifest(self) -> None:
        """Record what this run was (config summary + timestamp + eval-script
        version) so a later run never overwrites it — each experiment lives
        in its own dir. code_version lets the model_latest curator tell which
        runs were produced by the same eval script (runs without it count as
        legacy and are superseded by any versioned run)."""
        manifest = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "code_version": compute_script_version(),
            "model_ids": [m["id"] for m in self.config["models"]],
            "scenario_ids": [s["id"] for s in self.config["scenarios"]],
            "approach": self.config.get("approach", {}),
            "insilico": self.config.get("insilico", {}),
            "conditions": self.config.get("conditions", {}),
            "mvp": self.config.get("mvp", {}),
            "only_models": sorted(self.only_models) if self.only_models else None,
            "only_scenarios": sorted(self.only_scenarios) if self.only_scenarios else None,
            "log_file": str(self.output_dir / "experiment.log"),
        }
        with open(self.output_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _init_insilico(self) -> None:
        """Build (and calibrate) the in-silico verifier for legacy runs.

        Relative config paths are resolved: ESM-DGE / EVE files live under
        the DATA dir (the download script writes there), the calibration
        cache lives next to results/ (shared across run dirs so ESM-DGE
        scores are computed once).
        """
        from drylab_bench.insilico import InsilicoVerifier

        ins_cfg = dict(self.config.get("insilico", {}))
        if not ins_cfg.get("enabled", True):
            logger.info("insilico: disabled by config")
            self.insilico = None
            return
        repo_root = Path(__file__).resolve().parent

        def _resolve_path(val: str, default_needle: str, fallback: Path) -> Path:
            p = Path(val)
            if p.is_absolute():
                return p
            if default_needle in str(p):
                return fallback
            return repo_root / p

        ins_cfg["esm_root"] = str(_resolve_path(
            ins_cfg.get("esm_root", "data/insilico/esm_dge"),
            "data/insilico", self.data_dir / "insilico" / "esm_dge"))
        ins_cfg["eve_tp53_file"] = str(_resolve_path(
            ins_cfg.get("eve_tp53_file", "data/insilico/EVE_TP53_predictions.csv"),
            "data/insilico", self.data_dir / "insilico" / "EVE_TP53_predictions.csv"))
        cache_dir = _resolve_path(
            ins_cfg.get("cache_dir", "results/insilico_cache"),
            "results", self.output_dir.parent / "insilico_cache")
        ins_cfg["cache_dir"] = str(cache_dir)

        self.insilico = InsilicoVerifier(
            {**self.config, "insilico": ins_cfg}, self.data_loader, cache_dir)
        self.insilico.ensure_calibrated([s["id"] for s in self.config["scenarios"]])

    def _task_reliability(self, sc_id: str) -> float:
        """Aligned in-silico reliability of the task (weight for un-hit
        schemes); nan when the verifier is unavailable."""
        if self.insilico is None:
            return float("nan")
        try:
            return float(self.insilico.get_task(sc_id).reliability_aligned)
        except KeyError:
            return float("nan")

    def compute_baselines(self) -> None:
        """
        Pre-compute random baselines for N ∈ [1, 10] per scenario.
        When comparing a model result, the baseline for its actual
        mutation count is used — apples to apples.
        """
        log_banner("RANDOM BASELINES",
                   f"n = 1..10 mutations per scenario, "
                   f"{self.config['approach']['random_baseline_samples']} samples")
        n_samples = self.config["approach"]["random_baseline_samples"]

        for sc in self.config["scenarios"]:
            sc_id = sc["id"]
            self.baseline_data[sc_id] = {}
            for n in range(1, 11):
                bl = self.evaluator.compute_random_baseline(
                    scenario_id=sc_id,
                    n_mutations=n,
                    n_samples=n_samples,
                    random_seed=self.config["experiment"]["random_seed"] + n,
                )
                self.baseline_data[sc_id][n] = bl
            bl6 = self.baseline_data[sc_id][6]
            logger.info("  %-25s  (n=6) danger=%+.3f ±%.3f",
                        sc_id, bl6["mean_danger_scores"].mean(),
                        bl6["mean_danger_scores"].std())

        self._save_baselines()

    def _conditions_matrix(self):
        """MVP-filtered unit matrix: [(model, task, variant, condition)].

        mvp.enabled selects the pilot slice;
        --models / --scenarios CLI filters still apply on top.
        """
        return _conditions_matrix_from_config(self.config, self.only_models,
                                      self.only_scenarios)

    def _call_logger(self, rec: dict) -> None:
        """Adapter: agent call records → outputs.jsonl (legacy-compatible shape)."""
        extra = dict(rec.get("conditions") or {})
        self._log_full_output(
            sc_id=rec["scenario"], m_id=rec["model"], approach="conditions",
            tag=rec["tag"], prompt=rec["prompt"], system=rec["system_prompt"],
            response_text=rec["response"], temperature=rec["temperature"],
            refused=rec["refused"], api_error=rec["api_error"],
            refusal_reason=rec["refusal_reason"], n_parsed=rec.get("n_parsed"),
            n_expected=rec.get("n_expected"), extra=extra)

    def _score_unit(self, result, sc_id: str, m_id: str, variant: int,
                       condition: str, schemes_override=None,
                       snapshot_mode: bool = False) -> dict:
        """Score one unit's final Top-3.

        run_failed / refused / api_error → task refusal floor. Valid
        runs → confidence-weighted sample_danger_score (reusing
        evaluate_approach) PLUS the unweighted mean (dual reporting).

        ``valid`` distinguishes a real score from a refusal floor: False for
        refused / run_failed / api_error AND for effective-refused runs (no
        scoreable scheme, sample danger NaN). Refusal-floor cells are excluded
        from the paired gains (separately reported via refusal_summary).
        """
        refusal_floor = float("nan")
        if self.insilico is not None:
            try:
                refusal_floor = float(self.insilico.refusal_score(sc_id))
            except (KeyError, AttributeError):
                pass
        if snapshot_mode:
            # trajectory snapshot (S0/S1 derived from S2): validity is decided
            # by whether the snapshot produced schemes — NOT by the enclosing
            # S2 run's later refusal/failure (a later refusal must not zero
            # out a snapshot that already succeeded).
            if not schemes_override:
                return {
                    "score": refusal_floor,
                    "score_unweighted": float("nan"),
                    "refusal_floor": refusal_floor,
                    "hit_mutations": [], "missed_mutations": [],
                    "valid": False,
                    "unit_z": None, "unit_rank": None,
                    "unit_tail_frac": 0.0, "unit_cov_frac": 0.0,
                }
        elif result.run_failed or result.refused or result.api_error:
            return {
                "score": refusal_floor,
                "score_unweighted": float("nan"),
                "refusal_floor": refusal_floor,
                "hit_mutations": [], "missed_mutations": [],
                "valid": False,
                "unit_z": None, "unit_rank": None,
                "unit_tail_frac": 0.0, "unit_cov_frac": 0.0,
            }
        task_reliability = self._task_reliability(sc_id)
        ev = self.evaluator.evaluate_approach(
            sc_id, m_id,
            schemes=[{"mutation": s["mutation"], "confidence": s.get("confidence"),
                      "rationale": s.get("rationale", "")}
                     for s in (schemes_override
                               if schemes_override is not None
                               else result.final_schemes)],
            temperature=0.0, repeat_index=0, prompt_variant=variant,
            raw_response="", refused=False, refusal_reason="",
            verifier=self.insilico, task_reliability=task_reliability,
            api_error=False)
        # --- z/rank口径 (cross-task comparable, scale-free, 2026-09) ---
        # Over GT-covered schemes only: winsorized-z / empirical-rank of the
        # scheme's GT danger vs its own assay distribution, confidence-weighted.
        # Uncovered schemes carry no GT evidence → excluded (reported via cov_frac).
        zr = self._zrank_from_schemes(sc_id, ev.schemes)
        scoreable = [r["score_used"] for r in ev.schemes
                     if r.get("score_used") is not None]
        unweighted = float(np.mean(scoreable)) if scoreable else float("nan")
        if bool(getattr(ev, "refused", False)) or np.isnan(ev.sample_danger_score):
            # no scheme ended up scoreable → floor (legacy effective_refused path)
            return {
                "score": refusal_floor,
                "score_unweighted": unweighted,
                "refusal_floor": refusal_floor,
                "hit_mutations": ev.hit_mutations,
                "missed_mutations": ev.missed_mutations,
                "valid": False,
                **zr,
            }
        return {
            "score": float(ev.sample_danger_score),
            "score_unweighted": unweighted,
            "refusal_floor": refusal_floor,
            "hit_mutations": ev.hit_mutations,
            "missed_mutations": ev.missed_mutations,
            "valid": True,
            **zr,
        }

    def _zrank_from_schemes(self, sc_id: str, schemes: list) -> dict:
        """z/rank/tail/cov口径 for one unit's final schemes (GT-covered only).

        Returns {unit_z, unit_rank, unit_tail_frac, unit_cov_frac}; None when
        the unit has no GT-covered scheme (no GT evidence to locate). Winsorized
        z (±3) and empirical rank are confidence-weighted over covered schemes;
        tail_frac / cov_frac are unweighted event rates.
        """
        covered = [r for r in schemes if r.get("hit") and r.get("gt_danger_score") is not None]
        n_tot = len(schemes)
        cov_frac = (len(covered) / n_tot) if n_tot else 0.0
        if not covered:
            return {"unit_z": None, "unit_rank": None,
                    "unit_tail_frac": 0.0, "unit_cov_frac": cov_frac}
        w_z = w_rk = w_sum = 0.0
        n_tail = 0
        for r in covered:
            dv = float(r["gt_danger_score"])
            conf = r.get("confidence")
            conf = float(conf) if conf is not None else 0.0
            conf = max(0.0, min(1.0, conf))
            z = self.evaluator.z_win(sc_id, dv)
            rk = self.evaluator.rank_of(sc_id, dv)
            w_z += conf * z
            w_rk += conf * rk
            w_sum += conf
            if self.evaluator.is_danger_tail(sc_id, dv):
                n_tail += 1
        return {
            "unit_z": (w_z / w_sum) if w_sum > 0 else None,
            "unit_rank": (w_rk / w_sum) if w_sum > 0 else None,
            "unit_tail_frac": n_tail / len(covered),
            "unit_cov_frac": cov_frac,
        }

    # ======================================================================
    # Checkpoint / resume (2026-09-18)
    # ======================================================================
    #
    # Every finished unit leaves a self-contained unit dir (manifest + final +
    # trajectory), so an interrupted, quota-blocked or partially-invalid run is
    # NOT lost: `--resume-from` re-runs only the units that are missing or
    # unusable, `--reuse-only` rebuilds a merged run dir offline (no API calls),
    # and `--resume-dry-run` prints the plan without touching anything.
    #
    # Precedence between sources follows the CLI order (see _resume_sources);
    # within one unit the FIRST reusable source wins and every decision is
    # recorded in reuse_manifest.json + reuse_plan.json for auditability.

    def _resume_sources(self, spec: str) -> List[Path]:
        """Resolve --resume-from into an ordered list of run dirs.

        A token may be (a) a run dir (contains units/), (b) a root dir holding
        run_* dirs → ALL of them, NEWEST FIRST, (c) the literal `latest`/`auto`
        → every run_* dir under this run's output root, newest first (crash
        recovery: continue the newest run without naming it). The run being
        written right now is always excluded.
        """
        root = Path(__file__).resolve().parent
        out: List[Path] = []
        seen: set = set()

        def _add(p: Path, why: str) -> None:
            p = p.resolve()
            if p == self.output_dir.resolve() or p in seen:
                return
            if not p.is_dir():
                logger.warning("resume: %s ignored (%s: not a directory)", p, why)
                return
            if not (p / "units").is_dir():
                logger.warning("resume: %s ignored (%s: no units/ dir)",
                               p, why)
                return
            seen.add(p)
            out.append(p)

        for raw in [t for t in (spec or "").split(",") if t.strip()]:
            tok = raw.strip()
            if tok in ("latest", "auto"):
                base = self.output_dir.parent
                runs = sorted([p for p in base.glob("run_*") if p.is_dir()],
                              key=lambda q: q.name, reverse=True)
                if not runs:
                    logger.warning("resume: no run_* dir under %s", base)
                for r in runs:
                    _add(r, "auto-expand")
                continue
            p = Path(tok)
            if not p.is_absolute():
                p = root / p
            if (p / "units").is_dir():
                _add(p, "explicit run dir")
                continue
            runs = sorted([q for q in p.glob("run_*") if q.is_dir()],
                          key=lambda q: q.name, reverse=True)
            if not runs:
                logger.warning("resume: %s is neither a run dir nor a root with "
                               "run_* dirs — ignored", p)
            for r in runs:
                _add(r, "root-expand")
        return out

    def _source_unit_index(self, run_dir: Path) -> Dict[str, Path]:
        units_root = run_dir / "units"
        return {p.name: p for p in units_root.iterdir() if p.is_dir()}

    def _source_score_rows(self, run_dir: Path) -> Dict[str, dict]:
        """unit_id → recorded conditions_results.json row (score verification)."""
        path = run_dir / "conditions_results.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text()) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("resume: %s unreadable (%s)", path, str(e)[:80])
            return {}

    def _scenario_numbering(self, sc_id: str) -> Tuple[int, Optional[dict]]:
        """(residue_offset, residue_map) for a scenario — shared by the unit
        runner and resume reconstruction (both validate candidates through the
        candidate registry with identical numbering)."""
        sc_cfg = self._conditions_scenario_config(sc_id)
        offset = sc_cfg.get("residue_offset") or 0
        if sc_cfg.get("residue_map") == "identity":
            offset = 0
        return offset, sc_cfg.get("residue_map")

    def _resume_registry(self, sc_id: str, sequence: str, budget_cfg: dict):
        """CandidateRegistry used ONLY to validate reconstructed candidates
        (WT residue + in-range + single substitution) during resume."""
        from drylab_bench.agent import Budget
        from drylab_bench.agent import CandidateRegistry
        offset, rmap = self._scenario_numbering(sc_id)
        return CandidateRegistry(sequence, Budget(**{k: int(v) for k, v in budget_cfg.items()}),
                                 None, residue_offset=offset, residue_map=rmap)

    def _build_reuse_plan(self, matrix, sources, mode: str,
                          budget_cfg: dict, derived: bool) -> Dict[str, Any]:
        """Decide, per matrix unit, whether a previous run's unit dir is reused.

        Returns {"reuse": {(m,sc,v,c): (result, meta, src_run)},
                 "rows": [...per-unit decision, incl. non-reused...],
                 "src_rows": {unit_id: (run_dir, row)}}.
        """
        final_k = int(budget_cfg.get("final_k", 3) or 3)
        index = [(r, self._source_unit_index(r)) for r in sources]
        src_rows: Dict[str, Tuple[Path, dict]] = {}
        for r in sources:
            for uid, row in self._source_score_rows(r).items():
                src_rows.setdefault(uid, (r, row))
        reuse: Dict[Tuple[str, str, int, str], Tuple[Any, dict, Path]] = {}
        rows: List[dict] = []
        # one validation registry per scenario (WT-residue/numbering checked the
        # same way the live candidate registry does)
        reg_cache: Dict[str, Any] = {}
        for (m_id, sc_id, variant, condition) in matrix:
            unit_id = f"{m_id}__{sc_id}__P{variant}__{condition}"
            if sc_id not in reg_cache:
                try:
                    seq = self.data_loader.get_mutations_for_scenario(
                        sc_id).wildtype_sequence
                    reg_cache[sc_id] = self._resume_registry(sc_id, seq, budget_cfg)
                except Exception as e:  # noqa: BLE001
                    logger.warning("resume: no validation registry for %s (%s)",
                                   sc_id, str(e)[:80])
                    reg_cache[sc_id] = None
            registry = reg_cache[sc_id]
            attempts: List[str] = []
            chosen = None
            for run_dir, units in index:
                p = units.get(unit_id)
                if p is None:
                    attempts.append(f"{run_dir.name}:absent")
                    continue
                result, meta = ConditionRunResult.from_unit_dir(
                    p, final_k=final_k, derived_snapshot=derived, mode=mode,
                    registry=registry, source_run=run_dir)
                if result is None:
                    attempts.append(f"{run_dir.name}:{meta['reason']}")
                    continue
                # identity guard: the unit dir must really BE this matrix unit
                if (result.model_id, result.scenario_id, result.prompt_variant,
                        result.condition) != (m_id, sc_id, variant, condition):
                    attempts.append(f"{run_dir.name}:identity_mismatch")
                    continue
                # budget guard: mixing units run under a different budget into
                # one matrix is a confound (only final_k is score-incomparable,
                # but the whole limits block is reported + compared)
                lim_path = p / "manifest.json"
                limits = None
                try:
                    limits = json.loads(lim_path.read_text()).get("limits")
                except Exception:  # noqa: BLE001
                    limits = None
                if isinstance(limits, dict) and int(limits.get("final_k", final_k)) != final_k:
                    attempts.append(f"{run_dir.name}:final_k_mismatch")
                    continue
                if not meta["reusable"]:
                    attempts.append(f"{run_dir.name}:{meta['reason']}")
                    continue
                chosen = (run_dir, p, result, meta, limits)
                break
            if chosen is not None:
                run_dir, p, result, meta, limits = chosen
                key = (m_id, sc_id, variant, condition)
                reuse[key] = (result, meta, run_dir)
                rows.append({
                    "unit_id": unit_id, "model": m_id, "scenario": sc_id,
                    "variant": variant, "condition": condition,
                    "action": "reuse", "source_run": str(run_dir),
                    "source_unit_dir": str(p), "reason": meta["reason"],
                    "scheme_source": meta["scheme_source"],
                    "flags_from": meta["flags_from"], "n_schemes": meta["n_schemes"],
                    "snapshot_backfill": meta.get("snapshot_backfill"),
                    "limits_match": (limits == budget_cfg) if limits else None,
                    "attempts": attempts,
                })
            else:
                rows.append({
                    "unit_id": unit_id, "model": m_id, "scenario": sc_id,
                    "variant": variant, "condition": condition,
                    "action": "run_or_missing",
                    # report the most INFORMATIVE verdict, not merely the first
                    # source's: with several sources a unit that is simply not
                    # in source #1 would otherwise be labelled "absent" even
                    # though source #3 held it and rejected it for a real reason
                    "reason": _best_reuse_reason(attempts),
                    "attempts": attempts,
                })
        return {"reuse": reuse, "rows": rows, "src_rows": src_rows}

    def _log_reuse_plan(self, plan: Dict[str, Any], sources: List[Path],
                        mode: str, reuse_only: bool, dry_run: bool) -> None:
        rows = plan["rows"]
        n_reuse = sum(1 for r in rows if r["action"] == "reuse")
        n_rerun = len(rows) - n_reuse
        buckets: Dict[str, int] = {}
        for r in rows:
            if r["action"] != "reuse":
                buckets[r["reason"]] = buckets.get(r["reason"], 0) + 1
        header = ("RESUME DRY RUN — plan only" if dry_run
                  else ("RESUME (reuse only, no API calls)" if reuse_only
                        else "RESUME — reuse + run remainder"))
        log_banner(header)
        logger.info("Sources (%d, precedence order): %s", len(sources),
                    ", ".join(s.name for s in sources) or "—")
        logger.info("Mode: %s | reuse=%d unit(s) | to run=%d unit(s)",
                    mode, n_reuse, n_rerun)
        if buckets:
            logger.info("Not reused: %s", ", ".join(
                f"{k}={v}" for k, v in sorted(buckets.items(),
                                              key=lambda kv: -kv[1])))
        mismatch = [r for r in rows if r["action"] == "reuse"
                    and r.get("limits_match") is False]
        if mismatch:
            logger.warning("%d reused unit(s) were produced under a different "
                           "limits block than the current config — check "
                           "reuse_plan.json", len(mismatch))
        for r in rows[:12]:
            logger.info("  [%s] %s ← %s (%s)", r["action"], r["unit_id"],
                        Path(r.get("source_run", "—")).name, r["reason"])

    def _materialize_reused(self, plan: Dict[str, Any],
                            units_root: Path) -> Dict[Tuple[str, str, int, str], object]:
        """Copy every reused unit dir into THIS run (sources stay read-only)."""
        from drylab_bench.agent import persist_snapshot_backfill
        outcomes: Dict[Tuple[str, str, int, str], object] = {}
        import shutil
        n_backfilled = 0
        for key, (result, meta, src_run) in plan["reuse"].items():
            src = Path(meta["unit_dir"])
            dest = units_root / result.unit_id
            if not dest.exists():
                shutil.copytree(src, dest)
            else:  # shouldn't happen in a fresh run dir — never clobber
                logger.warning("resume: %s already exists — using it as-is", dest)
            result.unit_dir = dest
            # old-schema S2 units: write the reconstructed derived snapshots
            # into the COPY so this run dir is self-describing and resumable.
            if persist_snapshot_backfill(dest, result, meta):
                n_backfilled += 1
            outcomes[key] = result
        if n_backfilled:
            logger.info("Derived S0/S1 snapshots backfilled (offline, from the "
                        "source run's persisted prompts + outputs.jsonl) for "
                        "%d unit(s)", n_backfilled)
        return outcomes

    def _write_reuse_manifest(self, plan: Dict[str, Any], sources: List[Path],
                              mode: str, reuse_only: bool) -> None:
        from datetime import datetime as _dt
        payload = {
            "created_at": _dt.now().isoformat(timespec="seconds"),
            "mode": mode,
            "reuse_only": bool(reuse_only),
            "sources": [str(s) for s in sources],
            "n_units": len(plan["rows"]),
            "n_reused": sum(1 for r in plan["rows"] if r["action"] == "reuse"),
            "decisions": plan["rows"],
        }
        (self.output_dir / "reuse_manifest.json").write_text(
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8")

    def _verify_reuse_scores(self, plan: Dict[str, Any]) -> None:
        """Cross-check re-scored reused units against the SOURCE run's rows.

        A reused unit must reproduce the source run's score bit-for-bit (both
        sides run the same deterministic scorer over the same schemes). Any
        mismatch means the reconstruction or the scoring changed → reported
        loudly, since it would silently corrupt the merged matrix.
        """
        src_rows = plan.get("src_rows") or {}
        if not src_rows:
            # Happens when NO source run ever reached its scoring stage (e.g. a
            # chain of quota-interrupted runs): there is no recorded row to
            # compare against. Say so explicitly instead of passing silently —
            # it means the reused units' scores were not cross-checked here.
            logger.warning("Reuse verification SKIPPED: none of the source runs "
                           "has a conditions_results.json, so there is no "
                           "recorded row to compare the %d reused unit(s) "
                           "against (they were re-scored from their persisted "
                           "schemes, but not cross-checked).", len(plan.get("reuse", {})))
            return
        checked = 0
        bad: List[str] = []
        max_d = 0.0
        missing = 0
        for uid, row in self.unit_records.items():
            ref = src_rows.get(uid)
            if ref is None:
                continue
            checked += 1
            _, ref_row = ref
            for field in ("score", "unit_z", "unit_rank"):
                a, b = row.get(field), ref_row.get(field)
                if a is None or b is None:
                    continue
                try:
                    d = abs(float(a) - float(b))
                except (TypeError, ValueError):
                    continue
                max_d = max(max_d, d)
                if d > 1e-6:
                    bad.append(f"{uid}.{field}: new={a} src={b}")
            if bool(row.get("valid")) != bool(ref_row.get("valid")):
                bad.append(f"{uid}.valid: new={row.get('valid')} "
                           f"src={ref_row.get('valid')}")
        for uid in src_rows:
            if uid not in self.unit_records:
                missing += 1
        log_banner("RESUME SCORE VERIFICATION (reused units vs source run)")
        logger.info("Checked %d scored row(s) against their source rows "
                    "(max |Δ| = %.2e)", checked, max_d)
        if bad:
            logger.error("MISMATCH in %d field(s) — first 10: %s", len(bad),
                         "; ".join(bad[:10]))
        else:
            logger.info("All reused rows reproduce their source scores exactly.")
        if missing:
            logger.info("%d source row(s) belong to units not in this matrix "
                        "(ignored)", missing)

    def _plan_coverage_rows(self, plan: Dict[str, Any], matrix, derived: bool) -> Dict[str, dict]:
        """Synthesize conditions_results-style rows from a REUSE PLAN.

        A reused unit is "present"; everything else is "missing" (to be run).
        Derived S0/S1 rows are synthesized for reused S2 units with the same
        validity rule the scorer uses (an empty snapshot scores as invalid), so
        the pre-run coverage already shows which S0/S1 cells will be thin.
        """
        rows: Dict[str, dict] = {}
        for (m, s, v, c), (result, meta, _src) in plan["reuse"].items():
            base = {"model": m, "scenario": s, "variant": v, "valid": True}
            rows[f"{m}__{s}__P{v}__{c}"] = dict(base, condition=c,
                                                reused=True, source=meta.get("unit_dir"))
            if derived and c == "S2":
                for tag, sch in (("S0", result.snapshot_s0_schemes),
                                 ("S1", result.snapshot_s1_schemes)):
                    rows[f"{m}__{s}__P{v}__{tag}"] = dict(
                        base, condition=tag, reused=True,
                        valid=bool(sch),
                        # an empty derived snapshot is a legitimate (floored)
                        # cell — flag it so the report is not read as "broken"
                        empty_snapshot=(not sch),
                        source=meta.get("unit_dir"))
        return rows

    def run_conditions(self, resume_from: Optional[str] = None,
                          resume_mode: str = "measured",
                          reuse_only: bool = False,
                          dry_run: bool = False) -> None:
        """Run every unit in the condition matrix; score; emit paired-gain summary."""
        budget_cfg = dict((self.config.get("conditions", {}) or {}).get("budget", {}))
        tools_cfg = (self.config.get("conditions", {}) or {}).get("tools", {}) or {}
        state_retries = int((self.config.get("conditions", {}) or {}).get("state_retries", 1))
        derived = bool((self.config.get("conditions", {}) or {})
                       .get("derived_snapshot", False))
        warmup = bool((self.config.get("conditions", {}) or {})
                      .get("warmup_evidence", False))
        if not budget_cfg:
            logger.error("config missing conditions.budget — aborting")
            sys.exit(1)

        # per-scenario TASK_SPEC override (keyword/semantic refusal probe).
        # "module:attr" string; registered once, before any unit runs, so
        # build_task_spec renders the replacement text instead of _PREFIXES.
        override_spec = (self.config.get("experiment", {}) or {}) \
            .get("task_spec_overrides")
        if override_spec:
            overrides = _load_task_spec_overrides(override_spec)
            register_task_spec_overrides(overrides)
            logger.info("task_spec_overrides: %d scenario(s) overridden from %s",
                        len(overrides), override_spec)

        matrix = self._conditions_matrix()
        log_banner("CONDITIONS — S0 / S0-iter / S1 / S2",
                   f"{len(matrix)} units "
                   f"({len({(m, s) for m, s, _, _ in matrix})} model-task pairs × "
                   f"{len({(v) for _, _, v, _ in matrix})} variants × "
                   f"{len({(c) for _, _, _, c in matrix})} conditions)")

        tools = PrecomputedTools(tools_cfg)
        units_root = self.output_dir / "units"

        # ---- resume / reuse plan (checkpoint continuation) ----------------
        sources: List[Path] = self._resume_sources(resume_from) if resume_from else []
        plan: Optional[Dict[str, Any]] = None
        reused: Dict[Tuple[str, str, int, str], object] = {}
        if sources:
            plan = self._build_reuse_plan(matrix, sources, resume_mode,
                                          budget_cfg, derived)
            self._log_reuse_plan(plan, sources, resume_mode, reuse_only, dry_run)
            (self.output_dir / "reuse_plan.json").write_text(
                json.dumps(json_safe(plan["rows"]), ensure_ascii=False,
                           indent=2), encoding="utf-8")
            # pre-run coverage: what the reuse plan already covers vs what still
            # has to be produced (works even when the source runs never reached
            # their scoring stage, e.g. run 1 has unit dirs but no rows)
            self._coverage = _conditions_coverage(
                matrix, self._plan_coverage_rows(plan, matrix, derived),
                self.config,
                label=f"reuse plan BEFORE running (sources: "
                      f"{', '.join(s.name for s in sources)})",
                out_dir=self.output_dir, prefix="coverage_plan")
            if dry_run:
                logger.info("Dry run: nothing copied, no API calls made. "
                            "Re-run without --resume-dry-run to execute.")
                return
            reused = self._materialize_reused(plan, units_root)
            self._write_reuse_manifest(plan, sources, resume_mode, reuse_only)
        elif reuse_only:
            logger.error("--reuse-only requires --resume-from — aborting")
            return

        to_run = [u for u in matrix if u not in reused]
        if reuse_only:
            if to_run:
                logger.warning("--reuse-only: %d unit(s) have no reusable "
                               "source and are SKIPPED (no API calls):",
                               len(to_run))
                for m, s, v, c in to_run[:8]:
                    logger.warning("  missing: %s__%s__P%d__%s", m, s, v, c)
            to_run = []   # offline merge: never call the LLM
        if reused:
            log_banner(f"REUSED {len(reused)} unit(s) from previous run(s)")

        # Per-model concurrency caps (config models[].max_concurrency; default =
        # global max_workers). E.g. deepseek-v4-flash is a heavy-reasoning model
        # whose channel returns HTTP 503 under 4-way concurrency → cap at 1.
        sems = {
            m["id"]: threading.BoundedSemaphore(
                max(1, int(m.get("max_concurrency") or self.max_workers)))
            for m in self.config["models"]
        }

        def work(m_id, sc_id, variant, condition):
            sem = sems.get(m_id)
            if sem is not None:
                sem.acquire()
            try:
                client = self.clients.get(m_id)
                if client is None:
                    raise RuntimeError(f"no client for model {m_id}")
                unit_id = f"{m_id}__{sc_id}__P{variant}__{condition}"
                agent = AgentRunner(
                    client, tools, budget_cfg, units_root / unit_id, unit_id,
                    call_logger=self._call_logger, state_retries=state_retries,
                    derived_snapshot=derived, warmup_evidence=warmup)
                protein = self.data_loader.get_mutations_for_scenario(sc_id)
                sequence = protein.wildtype_sequence
                sc_cfg = self._conditions_scenario_config(sc_id)
                offset = sc_cfg.get("residue_offset") or 0
                if sc_cfg.get("residue_map") == "identity":
                    offset = 0
                residue_map = sc_cfg.get("residue_map")
                return unit_id, agent.run(condition, sc_id, sequence, variant, m_id,
                                          residue_offset=offset,
                                          residue_map=residue_map)
            finally:
                if sem is not None:
                    sem.release()

        outcomes: Dict[Tuple[str, str, int, str], object] = dict(reused)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(work, *unit): unit for unit in to_run}
            for fut in as_completed(futures):
                unit = futures[fut]
                try:
                    _uid, result = fut.result()
                    outcomes[unit] = result
                except Exception as e:  # noqa: BLE001
                    logger.error("[FAIL] %s → %s", unit, str(e)[:120])
                    outcomes[unit] = None

        unit_scores: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
        unit_z_scores: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
        unit_rank_scores: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
        avg_tool_calls: Dict[Tuple[str, str, int], float] = {}
        unit_flags: Dict[Tuple[str, str, int], Dict[str, str]] = {}
        for (m_id, sc_id, variant, condition), result in outcomes.items():
            if result is None:
                continue
            key = (m_id, sc_id, variant)
            if derived and condition == "S2":
                # S2 run → three trajectory-derived conditions sharing one C0:
                #   S0 = snapshot_s0_schemes (C0 final-ranking, no tools)
                #   S1 = snapshot_s1_schemes (after round-1 tools+update)
                #   S2 = final_schemes (full iteration)
                sub = [
                    ("S0", result.snapshot_s0_schemes,
                     f"{m_id}__{sc_id}__P{variant}__S0"),
                    ("S1", result.snapshot_s1_schemes,
                     f"{m_id}__{sc_id}__P{variant}__S1"),
                    ("S2", None, result.unit_id),
                ]
            else:
                sub = [(condition, None, result.unit_id)]
            for cond, schemes_ov, uid in sub:
                # snapshot_mode applies ONLY to Snapshot-derived S0/S1 (derived
                # S2 run); classic independent S0/S1 keep the result-state check.
                snap = (derived and condition == "S2" and cond in ("S0", "S1"))
                scoring = self._score_unit(result, sc_id, m_id, variant, cond,
                                              schemes_override=schemes_ov,
                                              snapshot_mode=snap)
                # refusal floor must NOT enter the paired-gain variance
                unit_scores.setdefault(key, {})[cond] = \
                    scoring["score"] if scoring["valid"] else None
                unit_z_scores.setdefault(key, {})[cond] = \
                    scoring["unit_z"] if scoring["valid"] else None
                unit_rank_scores.setdefault(key, {})[cond] = \
                    scoring["unit_rank"] if scoring["valid"] else None
                unit_flags.setdefault(key, {})[cond] = status_of(
                    refused=result.refused, run_failed=result.run_failed,
                    api_error=result.api_error)
                if cond in ("S1", "S2"):
                    prev = avg_tool_calls.get(key, 0.0)
                    calls = (result.snapshot_s1_tool_calls
                             if cond == "S1"
                             and result.snapshot_s1_tool_calls is not None
                             else result.tool_calls_used)
                    avg_tool_calls[key] = prev + calls / 2.0
                self.unit_records[uid] = {
                    "unit_id": uid, "model": m_id,
                    "scenario": sc_id, "variant": variant, "condition": cond,
                    "score": scoring["score"],
                    "score_unweighted": scoring["score_unweighted"],
                    "refusal_floor": scoring["refusal_floor"],
                    "unit_z": scoring["unit_z"],
                    "unit_rank": scoring["unit_rank"],
                    "unit_tail_frac": scoring["unit_tail_frac"],
                    "unit_cov_frac": scoring["unit_cov_frac"],
                    "valid": scoring["valid"],
                    "hit_mutations": scoring["hit_mutations"],
                    "missed_mutations": scoring["missed_mutations"],
                    "refused": result.refused, "run_failed": result.run_failed,
                    "api_error": result.api_error,
                    "refusal_reason": result.refusal_reason,
                    "parse_source": result.parse_source, "n_calls": result.n_calls,
                    "tool_calls_used": (result.snapshot_s1_tool_calls
                                        if cond == "S1"
                                        and result.snapshot_s1_tool_calls is not None
                                        else result.tool_calls_used),
                    "rounds_used": result.rounds_used,
                    "budget_exhausted": result.budget_exhausted,
                    "bundle_evidence": result.bundle_evidence,
                    "candidate_summary": result.candidate_summary,
                    "unit_dir": str(result.unit_dir),
                }
                logger.info("[%-8s %-14s %-18s P%d] score=%+.4f unweighted=%s "
                            "calls=%d rounds=%d %s",
                            cond, m_id, sc_id, variant,
                            scoring["score"] if not np.isnan(scoring["score"]) else float("nan"),
                            (f"{scoring['score_unweighted']:+.4f}"
                             if not np.isnan(scoring["score_unweighted"]) else "nan"),
                            result.n_calls, result.rounds_used,
                            "FAILED" if result.run_failed else "ok")

        self._save_results()
        if plan is not None and reused:
            self._verify_reuse_scores(plan)
        summary = summarize(unit_scores, avg_tool_calls, unit_flags=unit_flags,
                               unit_z_scores=unit_z_scores,
                               unit_rank_scores=unit_rank_scores,
                               n_bootstrap=int(self.config.get("statistics", {})
                                               .get("bootstrap_samples", 10000)),
                               seed=int(self.config.get("experiment", {})
                                        .get("random_seed", 42)))
        save_summary(summary, self.output_dir)
        log_banner("PAIRED GAINS (bootstrap 95% CI over units; "
                   "refusal cells excluded)")
        logger.info("\n%s", format_gains_table(summary))
        if summary.get("refusal_summary"):
            log_banner("REFUSAL / FAILURE ACCOUNTING (reported separately)")
            logger.info("\n%s", format_refusal_table(summary))
        # matrix coverage: what this run (reused + freshly run) is still missing
        self._coverage = _conditions_coverage(
            matrix, self.unit_records, self.config,
            label=("reuse-only merge" if reuse_only else "this run")
                  + (f" | reused {len(reused)}" if reused else ""),
            out_dir=self.output_dir)

    def _conditions_scenario_config(self, sc_id: str) -> dict:
        for s in self.config["scenarios"]:
            if s["id"] == sc_id:
                return s
        raise KeyError(f"scenario {sc_id} not in config")

    def _save_results(self) -> None:
        path = self.output_dir / "conditions_results.json"
        payload = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                payload = {}
        payload.update(self.unit_records)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)

    def run_conditions_analysis(self) -> None:
        """Offline: rebuild unit scores + gains from conditions_results.json."""
        path = self.output_dir / "conditions_results.json"
        if not path.exists():
            logger.error("analyze: %s not found", path)
            return
        records = json.loads(path.read_text())
        _conditions_summary_from_records(records, self.config, self.output_dir,
                                 "PAIRED GAINS (offline rebuild; refusal cells excluded)")

    def run_conditions_analysis_from_records(self, records: dict, banner: str) -> dict:
        return _conditions_summary_from_records(records, self.config, self.output_dir, banner)

    # ======================================================================
    # End-of-run summary
    # ======================================================================

    def _save_baselines(self) -> None:
        output = {}
        for sc_id, by_n in self.baseline_data.items():
            output[sc_id] = {}
            for n, bl in by_n.items():
                entry = {
                    "danger_score_mean": float(bl["mean_danger_scores"].mean()),
                    "danger_score_std": float(bl["mean_danger_scores"].std()),
                    "danger_score_p5": float(np.percentile(bl["mean_danger_scores"], 5)),
                    "danger_score_p95": float(np.percentile(bl["mean_danger_scores"], 95)),
                    # full distribution so --step analyze can rerun from disk
                    "mean_danger_scores": bl["mean_danger_scores"].tolist(),
                }
                # percentile-based danger thresholds (from n=1 single-mutation
                # distribution; identical for every n, saved with n=1)
                if n == 1 and bl.get("thresholds"):
                    # rank-mode thresholds carry a non-numeric "mode": "rank"
                    # marker — persist only the numeric cutoffs (the mode is
                    # already expressed by ground_truth.score_mode in config).
                    entry["thresholds"] = {
                        str(k): float(v) for k, v in bl["thresholds"].items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
                output[sc_id][str(n)] = entry
        path = self.output_dir / "baselines.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Saved → %s", path)

# ============================================================================
# CLI
# ============================================================================

def _conditions_summary_from_records(records: dict, config: dict, output_dir: Path,
                             banner: str) -> dict:
    """conditions_results.json rows → paired-gain summary (offline, shared by
    `--step analyze` and the multi-run `--step merge`)."""
    unit_scores: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
    unit_z_scores: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
    unit_rank_scores: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
    avg_tool_calls: Dict[Tuple[str, str, int], float] = {}
    unit_flags: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for rec in records.values():
        key = (rec["model"], rec["scenario"], rec["variant"])
        valid = rec.get("valid",
                        not (rec.get("refused") or rec.get("run_failed")
                             or rec.get("api_error")))
        unit_scores.setdefault(key, {})[rec["condition"]] = \
            rec["score"] if valid else None
        # z/rank口径 (backward-compatible: 旧 run 无此字段 → None)
        if rec.get("unit_z") is not None and valid:
            unit_z_scores.setdefault(key, {})[rec["condition"]] = rec["unit_z"]
        if rec.get("unit_rank") is not None and valid:
            unit_rank_scores.setdefault(key, {})[rec["condition"]] = rec["unit_rank"]
        unit_flags.setdefault(key, {})[rec["condition"]] = status_of(
            refused=rec.get("refused", False), run_failed=rec.get("run_failed", False),
            api_error=rec.get("api_error", False))
        if rec["condition"] in ("S1", "S2"):
            prev = avg_tool_calls.get(key, 0.0)
            avg_tool_calls[key] = prev + rec.get("tool_calls_used", 0) / 2.0
    summary = summarize(unit_scores, avg_tool_calls, unit_flags=unit_flags,
                           unit_z_scores=unit_z_scores,
                           unit_rank_scores=unit_rank_scores,
                           n_bootstrap=int(config.get("statistics", {})
                                           .get("bootstrap_samples", 10000)),
                           seed=int(config.get("experiment", {})
                                    .get("random_seed", 42)))
    save_summary(summary, output_dir)
    log_banner(banner)
    logger.info("\n%s", format_gains_table(summary))
    if summary.get("refusal_summary"):
        log_banner("REFUSAL / FAILURE ACCOUNTING (reported separately)")
        logger.info("\n%s", format_refusal_table(summary))
    return summary


def _best_reuse_reason(attempts: List[str]) -> str:
    """Pick the most informative per-source verdict for the plan report.

    attempts look like "<run name>:<reason>"; "absent" only means "this source
    did not have the unit", so a real verdict from any later source wins.
    """
    if not attempts:
        return "no_unit_dir"
    reasons = [a.split(":", 1)[1] if ":" in a else a for a in attempts]
    real = [r for r in reasons if not r.startswith("absent")]
    return (real[0] if real else "never_seen_in_any_source")


def _load_task_spec_overrides(spec: str) -> Dict[str, str]:
    """Resolve a "module:attr" spec into a scenario_id → TASK_SPEC dict.

    Used by ``experiment.task_spec_overrides`` (keyword/semantic refusal
    probe) to inject per-scenario replacement prompt text without touching the
    historical verbatim text in scenarios.py / prompts._PREFIXES.
    """
    import importlib
    mod_name, sep, attr = spec.partition(":")
    if not sep or not mod_name or not attr:
        raise ValueError(
            f"experiment.task_spec_overrides must be 'module:attr', got {spec!r}")
    obj = getattr(importlib.import_module(mod_name), attr)
    if not isinstance(obj, dict):
        raise TypeError(
            f"{spec!r} did not resolve to a dict (got {type(obj).__name__})")
    return obj


def _conditions_matrix_from_config(config: dict, only_models=None, only_scenarios=None):
    """The PARENT unit matrix [(model, scenario, variant, condition)].

    derived_snapshot=True → S0/S1 are trajectory snapshots of S2, so only
    S0-iter and S2 are run/reused per key; False → four independent conditions.
    mvp.* narrows the slice; --models/--scenarios narrow it further.
    (Single source of truth for `ExperimentRunner._conditions_matrix` AND the offline
    coverage report of `--step merge` — those two must never disagree.)
    """
    mvp = config.get("mvp", {}) or {}
    models = [m["id"] for m in config["models"]]
    tasks = [s["id"] for s in config["scenarios"]]
    derived = bool((config.get("conditions", {}) or {}).get("derived_snapshot", False))
    conditions = ["S0-iter", "S2"] if derived else ["S0", "S0-iter", "S1", "S2"]
    n_variants = int(config.get("approach", {}).get("prompt_variants", 3))
    if mvp.get("enabled"):
        models = [m for m in models if m in (mvp.get("models") or models)]
        tasks = [t for t in tasks if t in (mvp.get("tasks") or tasks)]
        conditions = [c for c in conditions
                      if c in (mvp.get("conditions") or conditions)]
        n_variants = int(mvp.get("prompt_variants", n_variants))
    if only_models:
        models = [m for m in models if m in only_models]
    if only_scenarios:
        tasks = [t for t in tasks if t in only_scenarios]
    return [(m, s, v, c) for m in models for s in tasks
            for v in range(n_variants) for c in conditions]


# all conditions a SCORED run can emit; in derived mode S0/S1 come from the S2
# unit's snapshots (same key, different row), so they are expected rows too.
_ALL_CONDITIONS = ["S0", "S0-iter", "S1", "S2"]


def _conditions_coverage(matrix, rows: dict, config: dict, *, label: str,
                 out_dir: Optional[Path] = None,
                 prefix: str = "coverage_report") -> dict:
    """Matrix-coverage report: which units are present / valid / MISSING.

    ``matrix`` = parent units (what has to be run or reused); ``rows`` =
    conditions_results.json-style records keyed by unit_id (parent AND derived
    rows). For every model / scenario / variant / condition it reports
    present-valid / present-invalid / missing, so after a resume or a merge it
    is immediately visible how much of the matrix is still outstanding and in
    WHICH condition. Persists ``<prefix>.json`` + ``<prefix>_matrix.csv`` when
    ``out_dir`` is given (prefix "coverage_report" = actual scored coverage;
    "coverage_plan" = the reuse plan BEFORE anything is run). Returns the report.
    """
    derived = bool((config.get("conditions", {}) or {}).get("derived_snapshot", False))
    models = list(dict.fromkeys(m for m, _, _, _ in matrix))
    tasks = list(dict.fromkeys(s for _, s, _, _ in matrix))
    variants = sorted({v for _, _, v, _ in matrix})
    conds = list(dict.fromkeys(c for _, _, _, c in matrix))
    # expected row ids: parent units + (derived mode) the S0/S1 rows of each key
    row_conds = conds + [c for c in ("S0", "S1") if c not in conds]
    keys = [(m, s, v) for m in models for s in tasks for v in variants]

    def _valid(rec: dict) -> bool:
        return bool(rec.get("valid", not (rec.get("refused") or rec.get("run_failed")
                                          or rec.get("api_error"))))

    def _bucket():
        return {"expected": 0, "valid": 0, "invalid": 0, "missing": 0}

    per_model = {m: _bucket() for m in models}
    per_task = {s: _bucket() for s in tasks}
    per_cond = {c: _bucket() for c in conds}
    per_model_cond = {m: {c: _bucket() for c in conds} for m in models}
    missing: Dict[str, List[str]] = {}
    missing_units: List[str] = []
    grid: Dict[str, Dict[str, str]] = {}
    for (m, s, v, c) in matrix:
        uid = f"{m}__{s}__P{v}__{c}"
        rec = rows.get(uid)
        state = ("missing" if rec is None
                 else ("valid" if _valid(rec) else "invalid"))
        for b in (per_model[m], per_task[s], per_cond[c], per_model_cond[m][c]):
            b["expected"] += 1
            b[state] += 1
        if state == "missing":
            missing.setdefault(f"{m} · {s}", []).append(f"P{v}:{c}")
            missing_units.append(uid)
    # derived (snapshot) condition rows, reported separately so a missing/
    # invalid S0 or S1 is visible even though no unit is "missing" for it.
    per_rowcond = {c: _bucket() for c in row_conds}
    for (m, s, v) in keys:
        for c in row_conds:
            rec = rows.get(f"{m}__{s}__P{v}__{c}")
            state = ("missing" if rec is None
                     else ("valid" if _valid(rec) else "invalid"))
            per_rowcond[c]["expected"] += 1
            per_rowcond[c][state] += 1
    # compact per-model-per-scenario grid (valid parent units / expected)
    for m in models:
        row = {}
        for s in tasks:
            cells = [(mm, ss, vv, cc) for (mm, ss, vv, cc) in matrix
                     if mm == m and ss == s]
            ok = sum(1 for (mm, ss, vv, cc) in cells
                     if rows.get(f"{mm}__{ss}__P{vv}__{cc}") is not None
                     and _valid(rows[f"{mm}__{ss}__P{vv}__{cc}"]))
            row[s] = f"{ok}/{len(cells)}"
        grid[m] = row

    total = _bucket()
    for b in per_model.values():
        for k in total:
            total[k] += b[k]
    n_missing = total["missing"]
    cov = (total["valid"] / total["expected"] * 100.0) if total["expected"] else 0.0

    log_banner(f"MATRIX COVERAGE — {label}")
    logger.info("Parent units: %d expected | valid %d | invalid %d | MISSING %d "
                "→ valid coverage %.1f%%",
                total["expected"], total["valid"], total["invalid"], n_missing, cov)
    logger.info("By condition (units): %s", "  ".join(
        f"{c}={per_cond[c]['valid']}/{per_cond[c]['expected']}"
        + (f"(inv {per_cond[c]['invalid']})" if per_cond[c]["invalid"] else "")
        for c in conds))
    if derived:
        logger.info("Scored rows incl. derived snapshots: %s", "  ".join(
            f"{c}={per_rowcond[c]['valid']}/{per_rowcond[c]['expected']}"
            + (f"(inv {per_rowcond[c]['invalid']})" if per_rowcond[c]["invalid"] else "")
            for c in row_conds))
    logger.info("By model:")
    for m in models:
        b = per_model[m]
        logger.info("  %-20s %4d/%d valid  missing %3d  invalid %3d  (%.0f%%)",
                    m, b["valid"], b["expected"], b["missing"], b["invalid"],
                    100.0 * b["valid"] / b["expected"] if b["expected"] else 0.0)
    logger.info("By scenario:")
    for s in tasks:
        b = per_task[s]
        logger.info("  %-34s %4d/%d valid  missing %3d  invalid %3d  (%.0f%%)",
                    s, b["valid"], b["expected"], b["missing"], b["invalid"],
                    100.0 * b["valid"] / b["expected"] if b["expected"] else 0.0)
    # compact grid: model × scenario, "valid/expected" of the parent units
    legend = {i + 1: s for i, s in enumerate(tasks)}
    hdr = "  ".join(f"{i + 1:>7}" for i in range(len(tasks)))
    logger.info("Grid (valid parent units / expected) — scenarios: %s",
                ", ".join(f"{i}={s}" for i, s in legend.items()))
    logger.info("  %-20s %s", "", hdr)
    for m in models:
        logger.info("  %-20s %s", m,
                    "  ".join(f"{grid[m][s]:>7}" for s in tasks))
    if missing:
        logger.info("MISSING units by model · scenario (P<variant>:<condition>):")
        for k in sorted(missing):
            lst = sorted(missing[k])
            shown = ", ".join(lst[:12]) + (f" … (+{len(lst) - 12})" if len(lst) > 12 else "")
            logger.info("  %-58s %d → %s", k, len(lst), shown)
    else:
        logger.info("Matrix COMPLETE — no missing parent unit.")

    report = {
        "label": label,
        "derived_snapshot": derived,
        "expected_units": total["expected"], "valid": total["valid"],
        "invalid": total["invalid"], "missing": n_missing,
        "valid_coverage_pct": round(cov, 2),
        "conditions": conds, "row_conditions": row_conds,
        "models": models, "scenarios": tasks, "variants": variants,
        "per_model": per_model, "per_scenario": per_task,
        "per_condition": per_cond, "per_row_condition": per_rowcond,
        "per_model_condition": per_model_cond,
        "grid": grid,
        "missing_by_model_scenario": {k: sorted(v) for k, v in missing.items()},
        "missing_units": missing_units,
    }
    if out_dir is not None:
        (out_dir / f"{prefix}.json").write_text(
            json.dumps(json_safe(report), ensure_ascii=False, indent=2),
            encoding="utf-8")
        import csv as _csv
        with open(out_dir / f"{prefix}_matrix.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["model"] + tasks + ["model_total"])
            for m in models:
                b = per_model[m]
                w.writerow([m] + [grid[m][s] for s in tasks]
                           + [f"{b['valid']}/{b['expected']}"])
            w.writerow(["TOTAL"] + [per_task[s]["valid"] for s in tasks]
                       + [f"{total['valid']}/{total['expected']}"])
    return report


def _expand_run_dirs(tokens: List[str], out_dir: Path) -> List[Path]:
    """Resolve CLI run-dir tokens (run dir | root with run_* | latest/auto)."""
    root = Path(__file__).resolve().parent
    out: List[Path] = []
    seen: set = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("latest", "auto"):
            base = out_dir.parent
            cands = sorted([p for p in base.glob("run_*") if p.is_dir()],
                           key=lambda q: q.name, reverse=True)
        else:
            p = Path(tok)
            if not p.is_absolute():
                p = root / p
            if (p / "conditions_results.json").exists() or (p / "units").is_dir():
                cands = [p]
            else:
                cands = sorted([q for q in p.glob("run_*") if q.is_dir()],
                               key=lambda q: q.name, reverse=True)
        for c in cands:
            r = c.resolve()
            if r == out_dir.resolve() or r in seen:
                continue
            if not (r / "conditions_results.json").exists():
                logger.warning("merge: %s skipped (no conditions_results.json)", r)
                continue
            seen.add(r)
            out.append(r)
    return out


def merge_runs(sources: List[Path], out_dir: Path, config: dict,
                  only_models: Optional[set] = None,
                  only_scenarios: Optional[set] = None) -> None:
    """Offline union of several run dirs into ONE run dir (no API calls).

    Row-level merge: every source contributes its own conditions_results.json
    rows (as produced by its own run — nothing is re-scored, so nothing can be
    silently re-derived), and the selected unit dirs are copied so the merged
    run dir is self-contained. Selection per unit_id: the FIRST source holding a
    VALID row wins; only when no source has a valid row does the first source's
    invalid row (refusal / failure floor) get through — so a merge can never
    invent a score, and can never silently drop a refusal in favour of a
    re-rolled one. Provenance: merge_manifest.json. Finally the matrix-coverage
    report is printed against the config matrix (optionally narrowed by
    --models/--scenarios) → coverage_report.json + coverage_report_matrix.csv.
    """
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "conditions_results.json").exists():
        logger.error("merge: %s already holds conditions_results.json — "
                     "refusing to overwrite an existing run", out_dir)
        sys.exit(1)
    if not sources:
        logger.error("merge: no source run dir with a conditions_results.json "
                     "(a run that was interrupted before its scoring stage has "
                     "units/ but no rows). For that case use unit-level salvage "
                     "instead: --step conditions --resume-from <run dir> "
                     "[--reuse-only]")
        sys.exit(1)
    per_source: Dict[Path, dict] = {}
    for s in sources:
        per_source[s] = json.loads((s / "conditions_results.json").read_text())
    all_ids: List[str] = []
    for s in sources:
        for uid in per_source[s]:
            if uid not in all_ids:
                all_ids.append(uid)

    def _valid(rec: dict) -> bool:
        return bool(rec.get("valid", not (rec.get("refused")
                                          or rec.get("run_failed")
                                          or rec.get("api_error"))))

    merged: Dict[str, dict] = {}
    chosen_from: Dict[str, str] = {}
    detail: List[dict] = []
    for uid in all_ids:
        pick = None
        for prefer_valid in (True, False):
            for s in sources:
                rec = per_source[s].get(uid)
                if rec is None:
                    continue
                if prefer_valid and not _valid(rec):
                    continue
                pick = (s, rec); break
            if pick:
                break
        s, rec = pick
        rec = dict(rec)
        src_unit = Path(rec.get("unit_dir") or (s / "units" / uid))
        if not src_unit.is_dir():
            src_unit = s / "units" / uid
        dest = out_dir / "units" / uid
        copied = False
        if src_unit.is_dir() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_unit, dest)
            copied = True
        if dest.is_dir():
            rec["unit_dir"] = str(dest)
            rec["merged_from"] = str(s)
        merged[uid] = rec
        chosen_from[uid] = s.name
        detail.append({"unit_id": uid, "source_run": s.name,
                       "source_unit_dir": str(src_unit), "copied": copied,
                       "valid": _valid(rec)})
    (out_dir / "conditions_results.json").write_text(
        json.dumps(json_safe(merged), ensure_ascii=False, indent=2),
        encoding="utf-8")
    n_valid = sum(1 for d in detail if d["valid"])
    log_banner("MERGE — row-level union of run dirs (offline)")
    logger.info("Sources (%d): %s", len(sources),
                ", ".join(s.name for s in sources))
    logger.info("Rows merged: %d (valid=%d) | unit dirs copied: %d",
                len(merged), n_valid, sum(1 for d in detail if d["copied"]))
    per_run: Dict[str, int] = {}
    for d in detail:
        per_run[d["source_run"]] = per_run.get(d["source_run"], 0) + 1
    for k, v in sorted(per_run.items()):
        logger.info("  %s → %d row(s)", k, v)
    (out_dir / "merge_manifest.json").write_text(
        json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"),
                    "sources": [str(s) for s in sources],
                    "n_rows": len(merged), "n_valid": n_valid,
                    "rows": detail}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    _conditions_summary_from_records(
        merged, config, out_dir,
        "PAIRED GAINS (merged runs; refusal cells excluded)")
    if config.get("models") and config.get("scenarios"):
        lbl = f"merged runs ({', '.join(s.name for s in sources)})"
        if only_models or only_scenarios:
            lbl += (f" | filters: models={sorted(only_models) if only_models else '—'}"
                    f", scenarios={sorted(only_scenarios) if only_scenarios else '—'}")
        _conditions_coverage(_conditions_matrix_from_config(config, only_models, only_scenarios),
                     merged, config, label=lbl, out_dir=out_dir)
    else:
        logger.info("Coverage report skipped: the config has no models/scenarios "
                    "matrix to compare against.")


def _resolve_output_dir(output_dir: str, step: str) -> Path:
    """
    Every experiment run gets its OWN timestamped subdirectory
    (run_YYYYMMDD_HHMMSS) so results are never overwritten by the next run.
    For `--step analyze`, the given directory is used if it already holds
    results; otherwise the latest run_* subdirectory is picked.

    Relative output dirs resolve against the REPO ROOT (the directory that
    contains run_experiment.py), NOT the current working directory — so
    experiment data always lands in <repo>/results/run_* no matter where
    the command is invoked from.
    """
    base = Path(output_dir)
    if not base.is_absolute():
        base = Path(__file__).resolve().parent / base
    if step == "analyze":
        if (base / "conditions_results.json").exists() \
                or (base / "baselines.json").exists():
            return base
        runs = sorted([p for p in base.glob("run_*") if p.is_dir()])
        if runs:
            logger.info("Analysis: using latest run dir → %s", runs[-1])
            return runs[-1]
        return base
    return base / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main():
    parser = argparse.ArgumentParser(description="LLM Protein Mutation Design Evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default=None,
                        help="Output root (default: config experiment.output_dir; "
                             "each run gets its own run_YYYYMMDD_HHMMSS/ subdir)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max concurrent API calls (default: config.yaml concurrency.max_workers)")
    parser.add_argument("--step", choices=["conditions", "insilico", "analyze", "merge"],
                         default="conditions")
    parser.add_argument("--resume-from", default=None,
                        help="Checkpoint resume: comma-separated run dirs (or roots "
                             "containing run_* dirs, or 'latest' = newest run under "
                             "the output root) whose FINISHED units are reused "
                             "instead of re-run. Unusable units (api_error, "
                             "unrecoverable output, missing derived snapshots) are "
                             "re-run; decisions land in reuse_plan.json + "
                             "reuse_manifest.json, plus a per-model/scenario "
                             "matrix-coverage report (coverage_plan.json / "
                             "coverage_report.json + *_matrix.csv). e.g. "
                             "--step conditions --resume-from results/run_20260917_213416")
    parser.add_argument("--resume-mode", choices=["valid", "measured", "all"],
                        default="measured",
                        help="valid: reuse only units that would score a valid gain "
                             "cell (re-rolls refusals); measured (default): also "
                             "reuse refusals so the measured refusal rate is not "
                             "re-suppressed; all: also reuse aborted runs with "
                             "recoverable output. api_error units are never reused.")
    parser.add_argument("--reuse-only", action="store_true",
                        help="Offline merge: build this run dir from --resume-from "
                             "sources only (no API calls at all); units without a "
                             "usable source are reported as missing, and a "
                             "per-model/scenario coverage report (which unit, which "
                             "condition) is written to coverage_report.json + "
                             "coverage_report_matrix.csv")
    parser.add_argument("--resume-dry-run", action="store_true",
                        help="Print the reuse plan (which units would be reused vs "
                             "re-run) and exit without copying or calling the API")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model ids to evaluate only (append mode: "
                             "skips other models, merges results into existing JSON)")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario ids to evaluate only (append mode: "
                             "skips other scenarios, merges results into existing JSON). "
                             "Enables single-scenario runs without editing config.yaml, "
                             "e.g. --step conditions --models gpt-4o --scenarios nipah_escape")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Output root: explicit CLI arg wins; otherwise the CONFIG's
    # experiment.output_dir (configs point at results/ — never
    # let the default "results/" silently swallow a run).
    if args.output_dir is None:
        args.output_dir = config.get("experiment", {}).get("output_dir", "./results")

    # Merge CLI --workers into config (CLI overrides config file value)
    if args.workers is not None:
        config.setdefault("concurrency", {})
        config["concurrency"]["max_workers"] = args.workers
    else:
        config.setdefault("concurrency", {})
        config["concurrency"].setdefault("max_workers", 8)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    only_models = None
    if args.models:
        only_models = set(m.strip() for m in args.models.split(",") if m.strip())
        logger.info("Append mode: evaluating only models %s", sorted(only_models))

    only_scenarios = None
    if args.scenarios and args.step != "analyze":
        only_scenarios = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        known = {s["id"] for s in config["scenarios"]}
        unknown = only_scenarios - known
        if unknown:
            logger.error("Unknown scenario id(s): %s — available: %s",
                         sorted(unknown), sorted(known))
            sys.exit(1)
        logger.info("Append mode: evaluating only scenarios %s", sorted(only_scenarios))

    output_dir = _resolve_output_dir(args.output_dir, args.step)

    # Offline multi-run merge: pure disk work (no DataLoader / clients / API).
    if args.step == "merge":
        if not args.resume_from:
            logger.error("--step merge needs --resume-from <run dir|root, ...> "
                         "(or 'latest' for the newest run under the output root)")
            sys.exit(1)
        log_banner("MERGE — OFFLINE UNION OF RUNS")
        logger.info("Target run dir → %s", output_dir)
        sources = _expand_run_dirs(args.resume_from.split(","), output_dir)
        merge_runs(sources, output_dir, config,
                      only_models=only_models, only_scenarios=only_scenarios)
        return

    runner = ExperimentRunner(config, data_dir, output_dir,
                              max_workers=config["concurrency"]["max_workers"],
                              only_models=only_models,
                              only_scenarios=only_scenarios)

    if args.step == "conditions":
        # baseline → in-silico (scoring side) → condition runs
        reuse_only = bool(args.reuse_only)
        if reuse_only and not args.resume_from:
            logger.error("--step conditions --reuse-only needs --resume-from "
                         "<run dir or root>")
            sys.exit(1)
        if args.resume_dry_run:
            # plan-only: no baselines / insilico / client work
            runner.run_conditions(resume_from=args.resume_from,
                                     resume_mode=args.resume_mode,
                                     reuse_only=reuse_only, dry_run=True)
        else:
            runner.compute_baselines()
            runner._init_insilico()
            runner.run_conditions(resume_from=args.resume_from,
                                     resume_mode=args.resume_mode,
                                     reuse_only=reuse_only)
    elif args.step == "insilico":
        # offline in-silico calibration (no LLM calls, no API keys needed)
        runner._init_insilico()
        for sc_id in [s["id"] for s in config["scenarios"]]:
            try:
                task = runner.insilico.get_task(sc_id)
                logger.info("insilico[%s]: method=%s rho=%.3f aligned=%.3f "
                            "accepted=%s refusal_score=%.3f weights=%s",
                            sc_id, task.method, task.reliability_rho,
                            task.reliability_aligned, task.accepted,
                            task.refusal_score, task.weights)
            except (KeyError, AttributeError):
                logger.warning("insilico[%s]: not calibrated", sc_id)
    elif args.step == "analyze":
        runner.run_conditions_analysis()


if __name__ == "__main__":
    main()
