"""
Response parsing for the LLM + biological-tools iterative benchmark.

Two layers:
  1. Strict JSON extraction/parsing (final output, round state) with
     code-fence stripping and brace matching.
  2. Tolerant fallback: when JSON parsing fails, salvage mutations and
     confidences with the legacy parser (_canonicalize_mutation /
     parse_response) so a partially-formatted response is not
     silently thrown away (same philosophy as the legacy parsers).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from drylab_bench.scenarios import parse_response


def find_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced {...} JSON object from a response.

    Strips ```json fences, then scans for the longest balanced brace span
    (tolerant to prose around the object). Returns None when no object
    parses.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?|```", "", text)
    best = None
    start = 0
    while True:
        open_idx = cleaned.find("{", start)
        if open_idx == -1:
            break
        depth = 0
        in_str = False
        esc = False
        for i in range(open_idx, len(cleaned)):
            c = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[open_idx:i + 1]
                    try:
                        obj = json.loads(candidate)
                    except Exception:  # noqa: BLE001
                        break
                    if isinstance(obj, dict):
                        # keep the FIRST (outermost) successful object; later
                        # matches are nested objects that would overwrite it
                        if best is None:
                            best = obj
                    break
        start = open_idx + 1
    return best


def parse_final_output(
    response_text: str,
    final_k: int = 3,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], str]:
    """Parse a final-output response.

    Returns (obj, fallback_schemes, source):
      - obj: parsed JSON object when available
      - fallback_schemes: legacy-parser salvage (list of scheme dicts) when JSON
        parsing produced no usable candidates
      - source: "json" | "salvage" | "empty"
    The caller merges: JSON object takes precedence; salvage is a soft
    degradation path (trajectory records a parse fallback event).
    """
    obj = find_json_object(response_text)
    if obj is not None and isinstance(obj.get("final_candidates"), list) \
            and obj["final_candidates"]:
        return normalize_final(obj), None, "json"

    schemes = parse_response(response_text)
    if schemes:
        return None, schemes, "salvage"
    return None, None, "empty"


# Candidate-array keys some models use INSIDE a dict-shaped final_candidates
# envelope (e.g. gemini emits final_candidates: {stop_reason, remaining_budget,
# top_3_mutations: [...]}). Lift the inner list so candidates are recovered
# instead of dropped as a schema violation (tolerant-parse philosophy).
_CAND_LIST_KEYS = ("top_3_mutations", "top_mutations", "top_candidates",
                   "final_mutations", "candidates", "mutations")


def lift_final_candidates(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a dict-shaped ``final_candidates`` into a candidate LIST.

    If ``final_candidates`` is a dict carrying a non-empty candidate array —
    first under a known envelope key, then, as a general fallback, ANY value
    that is a list of candidate-like objects (each a dict with a ``mutation``
    field) — lift that list to the top level. The key-name fallback matters:
    models invent envelope keys (gemini has emitted ``top_3_mutations`` and
    ``top_k``). A dict with no usable candidate list (e.g. a plain rank→map)
    is left untouched — later validation treats it as empty (clean partial
    failure, never a crash).
    """
    fc = obj.get("final_candidates")
    if isinstance(fc, dict):
        lifted = None
        for key in _CAND_LIST_KEYS:
            v = fc.get(key)
            if isinstance(v, list) and v:
                lifted = v
                break
        if lifted is None:
            for v in fc.values():
                if (isinstance(v, list) and v
                        and all(isinstance(x, dict) and "mutation" in x
                                for x in v)):
                    lifted = v
                    break
        if lifted is not None:
            obj = dict(obj)
            obj["final_candidates"] = lifted
    return obj


def _canon(mutation: str) -> str:
    from drylab_bench.scenarios import _canonicalize_mutation
    return _canonicalize_mutation(mutation)


# --- tolerant normalization: lowercase enums, numeric strings, etc. ---------

def _norm_enum(value, mapping):
    if isinstance(value, str):
        return mapping.get(value.strip().lower(), value)
    return value


def _norm_number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def normalize_state(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a parsed round-state object before schema validation."""
    if not isinstance(obj, dict):
        return obj
    obj = lift_final_candidates(dict(obj))
    obj["decision"] = _norm_enum(obj.get("decision"),
                                 {"continue": "CONTINUE", "stop": "STOP"})
    obj["condition"] = _norm_enum(
        obj.get("condition"),
        {"s0": "S0", "s0-iter": "S0-iter", "s0iter": "S0-iter",
         "s1": "S1", "s2": "S2"})
    for u in (obj.get("candidate_updates") or []):
        if isinstance(u, dict):
            u["action"] = _norm_enum(
                u.get("action"),
                {"add": "add", "retain": "retain", "rerank": "rerank",
                 "reject": "reject", "reconsider": "reconsider",
                 "keep": "retain", "drop": "reject", "rank": "rerank",
                 "re-add": "reconsider", "readd": "reconsider"})
            u["current_confidence"] = _norm_number(u.get("current_confidence"))
    for r in (obj.get("active_candidate_ranking") or []):
        if isinstance(r, dict):
            r["confidence"] = _norm_number(r.get("confidence"))
            r["rank"] = _norm_number(r.get("rank"))
    if isinstance(obj.get("final_candidates"), list):
        for c in obj["final_candidates"]:
            if isinstance(c, dict):
                c["confidence"] = _norm_number(c.get("confidence"))
                c["rank"] = _norm_number(c.get("rank"))
    return obj


def normalize_final(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a parsed final-output object before schema validation."""
    if not isinstance(obj, dict):
        return obj
    obj = lift_final_candidates(dict(obj))
    obj["condition"] = _norm_enum(
        obj.get("condition"),
        {"s0": "S0", "s0-iter": "S0-iter", "s0iter": "S0-iter",
         "s1": "S1", "s2": "S2"})
    for c in (obj.get("final_candidates") or []):
        if isinstance(c, dict):
            c["confidence"] = _norm_number(c.get("confidence"))
            c["rank"] = _norm_number(c.get("rank"))
    return obj


def parse_round_state(
    response_text: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse one round's structured state JSON.

    Returns (obj, source) with source ∈ {"json", "none"}. No content salvage
    for round states: candidate actions must come from structured JSON
    (middleware retries once, then fails the run).
    """
    obj = find_json_object(response_text)
    if obj is not None and (
        "decision" in obj or "candidate_updates" in obj
        or "final_candidates" in obj
    ):
        return normalize_state(obj), "json"
    return None, "none"


def final_candidates_from_state(
    state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Extract the final-output payload from a STOP round state.

    Returns (candidates, stop_reason_raw): candidates = state["final_candidates"];
    stop_reason_raw = state.get("stop_reason"). The schema validator treats
    the embedded payload as the final output object.
    """
    cands = state.get("final_candidates") or []
    return cands, state.get("stop_reason")


def build_schemes_from_final(
    final_obj: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a final-output JSON into legacy-style scheme dicts for scoring
    reuse (evaluator.evaluate_approach expects mutation/confidence/
    rationale).

    Robustness guard: ``final_candidates`` must be a LIST of objects. A
    non-list (e.g. a dict, which some models emit) or non-object entries are
    treated as schema violations → no schemes (caller fails the run cleanly as
    partial output instead of crashing on ``c.get(...)``).
    """
    schemes = []
    cands = final_obj.get("final_candidates", [])
    if not isinstance(cands, list):
        return []
    for c in cands:
        if not isinstance(c, dict):
            continue
        schemes.append({
            "mutation": _canon(str(c.get("mutation", ""))),
            "confidence": c.get("confidence"),
            "rationale": str(c.get("decision_summary", ""))[:500],
            "raw_block": json.dumps(c, ensure_ascii=False)[:1000],
        })
    return schemes


def build_schemes_from_salvage(salvage: List[Dict[str, Any]],
                               registry=None) -> List[Dict[str, Any]]:
    """legacy-parser salvage → legacy-style scheme dicts (already in that shape).

    Audit fix (P1-4): dedupe by canonical mutation — a repeated mutation in a
    salvaged final would otherwise be confidence-weighted twice in scoring.
    First occurrence wins. When ``registry`` (a CandidateRegistry) is given,
    every salvaged mutation must also pass registry.parse_and_validate
    (WT residue + in-range + single substitution); invalid entries are dropped
    so the salvage path cannot bypass the same validation the JSON path gets.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for s in salvage:
        if not isinstance(s, dict):
            continue
        raw = str(s.get("mutation", "")).strip()
        if registry is not None:
            canon, err = registry.parse_and_validate(raw)
            if err:
                continue
        else:
            canon = _canon(raw)
            if not canon:
                continue
        if canon in seen:
            continue
        seen.add(canon)
        s["mutation"] = canon
        out.append(s)
    return out
