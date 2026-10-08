"""
JSON schemas and validation for the LLM + biological-tools iterative benchmark.

The SAME schemas are used by all four conditions — only field emptiness rules
differ per condition. Validators return
``(ok, errors)``; ``ok`` is True iff no HARD violations. Warnings (extra
fields, unknown enum values in optional slots) are soft and do not fail the
run — the middleware records them in the trajectory instead.
"""

from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "1.0"

ACTION_ENUM = {"add", "retain", "rerank", "reject", "reconsider"}
DECISION_ENUM = {"CONTINUE", "STOP"}
CONDITION_ENUM = {"S0", "S0-iter", "S1", "S2"}
TOOL_ENUM = {"evolution_msa", "sequence_plm", "structure_compatibility",
             "functional_annotation", "phenotype", "toxin"}
STOP_TYPE_ENUM = {"budget_exhausted", "model_stop"}

# Canonical per-scheme mutation token: <WT_AA><1-based_position><Mutant_AA>
_MUT_RE_STR = r"[A-Za-z]\d+[A-Za-z]"


# ============================================================================
# Final output schema
# ============================================================================

def validate_final_output(
    obj: Dict[str, Any],
    final_k: int = 3,
    condition: str = "S0",
) -> Tuple[bool, List[str]]:
    """Validate a parsed final output JSON object.

    Hard rules:
      - final_candidates present, exactly ``final_k`` entries
      - rank values are 1..final_k, unique, contiguous
      - mutation is a non-empty string
      - confidence is a float in [0, 1]
      - remaining_budget is a dict with numeric tool_calls/rounds (if present)
    """
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["final output is not a JSON object"]

    cands = obj.get("final_candidates")
    if not isinstance(cands, list):
        errors.append("final_candidates missing or not a list")
        cands = []
    if len(cands) != final_k:
        errors.append(f"expected exactly {final_k} final_candidates, got {len(cands)}")

    ranks = []
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            errors.append(f"final_candidates[{i}] is not an object")
            continue
        mut = c.get("mutation")
        if not isinstance(mut, str) or not mut.strip():
            errors.append(f"final_candidates[{i}].mutation missing/empty")
        rank = c.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, (int, float)) or int(rank) != rank:
            errors.append(f"final_candidates[{i}].rank invalid: {rank!r}")
        else:
            ranks.append(int(rank))
        conf = c.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            errors.append(f"final_candidates[{i}].confidence missing/invalid")
        elif not (0.0 <= float(conf) <= 1.0):
            errors.append(f"final_candidates[{i}].confidence out of [0,1]: {conf}")

    if ranks and sorted(ranks) != list(range(1, final_k + 1)):
        errors.append(f"ranks must be exactly 1..{final_k} unique, got {sorted(ranks)}")

    stop_reason = obj.get("stop_reason")
    if stop_reason is None:
        errors.append("stop_reason missing")
    elif isinstance(stop_reason, dict):
        stype = stop_reason.get("type")
        if stype not in STOP_TYPE_ENUM:
            errors.append(f"stop_reason.type not in {sorted(STOP_TYPE_ENUM)}: {stype!r}")

    budget = obj.get("remaining_budget")
    if budget is not None and isinstance(budget, dict):
        for k in ("tool_calls", "rounds"):
            v = budget.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                errors.append(f"remaining_budget.{k} must be numeric")

    return (len(errors) == 0, errors)


# ============================================================================
# Per-round structured state schema
# ============================================================================

def validate_round_state(
    obj: Dict[str, Any],
    condition: str,
    budget_remaining: int,
    planning_stage: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate one round's structured state JSON.

    ``condition`` gates next_tool_requests:
      - S0 / S0-iter: must be empty
      - S1: allowed only in the pre-evidence planning stage
      - S2: allowed while budget remains
    ``planning_stage`` is True only for S1's first call.
    """
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["round state is not a JSON object"]

    # A STOP payload may carry ONLY the embedded final fields
    # — in that case candidate_updates/ranking are optional.
    final_payload = (obj.get("decision") == "STOP"
                     and isinstance(obj.get("final_candidates"), list))

    decision = obj.get("decision")
    if decision not in DECISION_ENUM:
        errors.append(f"decision must be one of {sorted(DECISION_ENUM)}, got {decision!r}")

    updates = obj.get("candidate_updates")
    if not isinstance(updates, list):
        if not final_payload:
            errors.append("candidate_updates missing or not a list")
        updates = []
    seen_muts: set = set()
    for i, u in enumerate(updates):
        if not isinstance(u, dict):
            errors.append(f"candidate_updates[{i}] is not an object")
            continue
        action = u.get("action")
        if action not in ACTION_ENUM:
            errors.append(f"candidate_updates[{i}].action not in {sorted(ACTION_ENUM)}: {action!r}")
            continue
        mut = u.get("mutation")
        if not isinstance(mut, str) or not mut.strip():
            errors.append(f"candidate_updates[{i}].mutation missing/empty")
        else:
            if mut in seen_muts:
                errors.append(f"duplicate candidate_updates entry for {mut}")
            seen_muts.add(mut)
        conf = u.get("current_confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            errors.append(f"candidate_updates[{i}].current_confidence missing/invalid")
        elif not (0.0 <= float(conf) <= 1.0):
            errors.append(f"candidate_updates[{i}].current_confidence out of [0,1]: {conf}")
        if action in ("add", "reconsider") and not u.get("decision_rationale"):
            errors.append(f"candidate_updates[{i}] ({action}) requires decision_rationale")

    ranking = obj.get("active_candidate_ranking")
    if ranking is None:
        ranking = []  # optional: proposal rounds may legitimately omit it
    if not isinstance(ranking, list):
        if not final_payload:
            errors.append("active_candidate_ranking must be a list")
        ranking = []
    else:
        rranks = []
        for i, r in enumerate(ranking):
            if not isinstance(r, dict):
                errors.append(f"active_candidate_ranking[{i}] is not an object")
                continue
            rk = r.get("rank")
            if isinstance(rk, bool) or not isinstance(rk, (int, float)):
                errors.append(f"active_candidate_ranking[{i}].rank invalid")
            else:
                rranks.append(int(rk))
        if rranks and rranks != list(range(1, len(rranks) + 1)):
            errors.append(f"ranking ranks not contiguous 1..{len(rranks)}: {rranks}")

    requests = obj.get("next_tool_requests", [])
    if not isinstance(requests, list):
        errors.append("next_tool_requests must be a list")
        requests = []
    if condition in ("S0", "S0-iter") and requests:
        errors.append(f"{condition}: next_tool_requests must be empty")
    if condition == "S1" and requests and not planning_stage:
        errors.append("S1: next_tool_requests allowed only in the pre-evidence planning stage")
    if condition == "S2" and requests and budget_remaining <= 0:
        errors.append("S2: next_tool_requests not allowed with exhausted budget")
    for i, r in enumerate(requests):
        if not isinstance(r, dict):
            errors.append(f"next_tool_requests[{i}] is not an object")
            continue
        tool = r.get("tool")
        if tool not in TOOL_ENUM:
            errors.append(f"next_tool_requests[{i}].tool not in {sorted(TOOL_ENUM)}: {tool!r}")
        cands = r.get("candidates")
        if not isinstance(cands, list) or not cands or not all(isinstance(c, str) for c in cands):
            errors.append(f"next_tool_requests[{i}].candidates must be a non-empty list of strings")

    if decision == "STOP" and not obj.get("final_candidates"):
        # Final candidates must be present (or emitted
        # immediately) when stopping. The middleware tolerates a missing
        # payload by issuing ONE final-only request — this is a soft error.
        errors.append("decision=STOP without embedded final_candidates (middleware will request final JSON)")

    return (len(errors) == 0, errors)


# ============================================================================
# Schema documentation blocks embedded into prompts
# ============================================================================

FINAL_SCHEMA_SPEC = """\
Respond with ONE JSON object matching this schema (nothing else, no markdown fences):
{
  "schema_version": "1.0",
  "condition": "<S0 | S0-iter | S1 | S2>",
  "final_candidates": [
    {"rank": 1, "mutation": "<WT><POS><MUT>", "confidence": 0.87,
     "evidence_refs": ["E00003"], "decision_summary": "<= 2 sentences"},
    {"rank": 2, "mutation": "<WT><POS><MUT>", "confidence": 0.73,
     "evidence_refs": [], "decision_summary": "<= 2 sentences"},
    {"rank": 3, "mutation": "<WT><POS><MUT>", "confidence": 0.61,
     "evidence_refs": [], "decision_summary": "<= 2 sentences"}
  ],
  "stop_reason": {"type": "budget_exhausted | model_stop", "detail": "one sentence"},
  "remaining_budget": {"tool_calls": 0, "rounds": 0}
}
Rules: exactly 3 final_candidates; ranks exactly 1,2,3; mutation in [WT][POS][MUT]
notation; confidence a number in [0.00, 1.00] — values are independent and do
NOT need to sum to 1."""

STATE_SCHEMA_SPEC = """\
Respond with ONE JSON object matching this schema (nothing else, no markdown fences):
{
  "schema_version": "1.0",
  "round": 2,
  "condition": "<S0-iter | S1 | S2>",
  "decision": "CONTINUE | STOP",
  "candidate_updates": [
    {"mutation": "<WT><POS><MUT>", "action": "add | retain | rerank | reject | reconsider",
     "current_confidence": 0.78,
     "supporting_evidence_ids": ["E00003"],
     "contradicting_evidence_ids": ["E00021"],
     "origin": "initial_pool | new_hypothesis_from_evidence",
     "trigger_evidence_ids": ["E00017"],
     "decision_rationale": "<= 2 sentences"}
  ],
  "evidence_assessment": [
    {"evidence_id": "E00017", "candidate": "<WT><POS><MUT>",
     "llm_reliability_assessment": "high | medium | low",
     "effect_on_candidate": "support | weak_against | against",
     "reason": "<= 2 sentences"}
  ],
  "evidence_conflicts": [
    {"candidate": "<WT><POS><MUT>", "evidence_ids": ["E00017", "E00021"],
     "conflict_type": "sequence_supports_structure_opposes",
     "resolution": "favor_sequence_with_caution",
     "reason": "<= 2 sentences"}
  ],
  "active_candidate_ranking": [
    {"rank": 1, "mutation": "<WT><POS><MUT>", "confidence": 0.78}
  ],
  "next_tool_requests": [
    {"tool": "evolution_msa | sequence_plm | structure_compatibility | functional_annotation | phenotype",
     "candidates": ["<WT><POS><MUT>"],
     "question": "one sentence",
     "reason_for_call": "one sentence"}
  ],
  "stop_assessment": {"should_stop": false, "reason": "one sentence",
                      "expected_value_of_additional_evidence": "high | medium | low"}
}
If decision is STOP, embed the final output object under the key
"final_candidates" (with stop_reason / remaining_budget) in the SAME JSON —
omit "candidate_updates" updates in that case is allowed."""


# ============================================================================
# Phase-split schemas (multi-round iterative protocol — "3 phases per round")
#
# Each iteration separates the decision points so the model must explicitly
# (a) decide whether / which tools to query, then (b) digest new evidence and
# update the candidates, then (c) decide continue-vs-STOP. Round 0 emits only
# the candidate pool. The model never has to do everything in one JSON object.
# ============================================================================

POOL_SCHEMA_SPEC = """\
Respond with ONE JSON object matching this schema (nothing else, no markdown fences):
{
  "schema_version": "1.0",
  "condition": "<S0 | S0-iter | S2>",
  "candidate_pool": [
    {"mutation": "<WT><POS><MUT>", "confidence": 0.80,
     "prediction_rationale": "<= 2 sentences"}
  ]
}
Rules: at most 10 candidate_pool entries; each mutation in [WT][POS][MUT]
notation; confidence a number in [0.00, 1.00] (independent, need not sum to 1).
Do NOT request tools here — only propose the initial candidate pool."""

TOOL_SELECT_SCHEMA_SPEC = """\
Respond with ONE JSON object matching this schema (nothing else, no markdown fences):
{
  "schema_version": "1.0",
  "condition": "S2",
  "round": 1,
  "next_tool_requests": [
    {"tool": "evolution_msa | sequence_plm | structure_compatibility | functional_annotation | phenotype",
     "candidates": ["<WT><POS><MUT>", ...],
     "question": "one sentence",
     "reason_for_call": "one sentence"}
  ]
}
Rules: request only tools you still need; each tool call may evaluate at most
10 active candidates; do not re-request a tool for a candidate already covered
(a repeated call returns no new information). If you need no further evidence,
output an EMPTY next_tool_requests array."""

SELF_REVIEW_SCHEMA_SPEC = """\
Respond with ONE JSON object matching this schema (nothing else, no markdown fences):
{
  "schema_version": "1.0",
  "condition": "S0-iter",
  "round": 1,
  "self_review": {
    "summary": "1-2 sentences — your current read of the candidate pool",
    "doubts": "1 sentence or empty — remaining uncertainties",
    "should_reevaluate": true
  }
}
Rules: S0-iter has no external tools — this reflection replaces the tool-selection
step and feeds the re-evaluation that follows."""

UPDATE_SCHEMA_SPEC = """\
Respond with ONE JSON object matching this schema (nothing else, no markdown fences):
{
  "schema_version": "1.0",
  "condition": "<S0-iter | S2>",
  "round": 1,
  "candidate_updates": [
    {"mutation": "<WT><POS><MUT>", "action": "add | retain | rerank | reject | reconsider",
     "current_confidence": 0.78,
     "supporting_evidence_ids": ["E00003"],
     "contradicting_evidence_ids": ["E00021"],
     "origin": "initial_pool | new_hypothesis_from_evidence",
     "trigger_evidence_ids": ["E00017"],
     "decision_rationale": "<= 2 sentences"}
  ],
  "active_candidate_ranking": [
    {"rank": 1, "mutation": "<WT><POS><MUT>", "confidence": 0.78}
  ],
  "evidence_assessment": [
    {"evidence_id": "E00017", "candidate": "<WT><POS><MUT>",
     "llm_reliability_assessment": "high | medium | low",
     "effect_on_candidate": "support | weak_against | against",
     "reason": "<= 2 sentences"}
  ],
  "thinking": "<= 3 sentences — the reasoning behind THIS round's changes",
  "doubts": "<= 1 sentence — remaining uncertainties"
}
Rules: if you add/reconsider, give decision_rationale; every supporting/
contradicting evidence id referenced in updates must appear in evidence_assessment."""

STOP_CHECK_SCHEMA_SPEC = """\
Respond with ONE JSON object matching this schema (nothing else, no markdown fences):
{
  "schema_version": "1.0",
  "condition": "<S0-iter | S2>",
  "round": 1,
  "decision": "CONTINUE | STOP",
  "decision_rationale": "1-2 sentences"
}
If decision is STOP, embed the final output under "final_candidates" in the SAME
object (with stop_reason / remaining_budget):
{
  ...,
  "decision": "STOP",
  "decision_rationale": "...",
  "final_candidates": [
    {"rank": 1, "mutation": "<WT><POS><MUT>", "confidence": 0.87,
     "evidence_refs": ["E00003"], "decision_summary": "<= 2 sentences"},
    {"rank": 2, "mutation": "<WT><POS><MUT>", "confidence": 0.73,
     "evidence_refs": [], "decision_summary": "<= 2 sentences"},
    {"rank": 3, "mutation": "<WT><POS><MUT>", "confidence": 0.61,
     "evidence_refs": [], "decision_summary": "<= 2 sentences"}
  ],
  "stop_reason": {"type": "budget_exhausted | model_stop", "detail": "one sentence"},
  "remaining_budget": {"tool_calls": 0, "rounds": 0}
}
Rules: exactly 3 final_candidates; ranks exactly 1,2,3; mutation in [WT][POS][MUT].
If decision is CONTINUE, do NOT include final_candidates."""


# ============================================================================
# Phase validators (return (ok, errors), same convention as the originals)
# ============================================================================

def _tok(m):
    return isinstance(m, str) and bool(m.strip())


def validate_pool(obj: Any, max_candidates: int = 10) -> Tuple[bool, List[str]]:
    errors = []
    cands = obj.get("candidate_pool") if isinstance(obj, dict) else None
    if not isinstance(cands, list):
        return False, ["candidate_pool missing or not a list"]
    if not cands:
        errors.append("candidate_pool empty")
    if len(cands) > max_candidates:
        errors.append(f"candidate_pool has {len(cands)} > {max_candidates} entries")
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            errors.append(f"candidate_pool[{i}] is not an object"); continue
        if not _tok(c.get("mutation")):
            errors.append(f"candidate_pool[{i}].mutation missing/empty")
        conf = c.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) \
                or not (0.0 <= float(conf) <= 1.0):
            errors.append(f"candidate_pool[{i}].confidence invalid: {conf!r}")
    return (len(errors) == 0, errors)


def validate_tool_select(obj: Any, condition: str, budget_remaining: int) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(obj, dict):
        return False, ["tool selection is not a JSON object"]
    if condition != "S2":
        errors.append(f"tool selection is S2-only, got {condition!r}")
    reqs = obj.get("next_tool_requests")
    if not isinstance(reqs, list):
        return False, ["next_tool_requests missing or not a list"]
    for i, r in enumerate(reqs):
        if not isinstance(r, dict):
            errors.append(f"next_tool_requests[{i}] is not an object"); continue
        if r.get("tool") not in TOOL_ENUM:
            errors.append(f"next_tool_requests[{i}].tool invalid: {r.get('tool')!r}")
        cands = r.get("candidates")
        if not isinstance(cands, list) or not cands or not all(isinstance(x, str) for x in cands):
            errors.append(f"next_tool_requests[{i}].candidates must be a non-empty list of strings")
    return (len(errors) == 0, errors)


def validate_self_review(obj: Any) -> Tuple[bool, List[str]]:
    if not isinstance(obj, dict):
        return False, ["self review is not a JSON object"]
    sr = obj.get("self_review")
    if not isinstance(sr, dict):
        return False, ["self_review missing or not an object"]
    return True, []


def validate_update(obj: Any, condition: str) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(obj, dict):
        return False, ["update is not a JSON object"]
    ups = obj.get("candidate_updates")
    if not isinstance(ups, list):
        errors.append("candidate_updates missing or not a list")
        ups = []
    seen: set = set()
    for i, u in enumerate(ups):
        if not isinstance(u, dict):
            errors.append(f"candidate_updates[{i}] is not an object"); continue
        if u.get("action") not in ACTION_ENUM:
            errors.append(f"candidate_updates[{i}].action invalid: {u.get('action')!r}"); continue
        m = u.get("mutation")
        if not _tok(m):
            errors.append(f"candidate_updates[{i}].mutation missing/empty")
        else:
            if m in seen:
                errors.append(f"duplicate candidate_updates entry {m}")
            seen.add(m)
        conf = u.get("current_confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) \
                or not (0.0 <= float(conf) <= 1.0):
            errors.append(f"candidate_updates[{i}].current_confidence invalid: {conf!r}")
        if u.get("action") in ("add", "reconsider") and not u.get("decision_rationale"):
            errors.append(f"candidate_updates[{i}] ({u['action']}) requires decision_rationale")
    ranking = obj.get("active_candidate_ranking")
    if ranking is not None:
        if not isinstance(ranking, list):
            errors.append("active_candidate_ranking must be a list")
        else:
            rranks = []
            for r in ranking:
                if isinstance(r, dict) and isinstance(r.get("rank"), (int, float)):
                    rranks.append(int(r["rank"]))
            if rranks and rranks != list(range(1, len(rranks) + 1)):
                errors.append(f"ranking not contiguous 1..{len(rranks)}: {rranks}")
    return (len(errors) == 0, errors)


def validate_stop_check(obj: Any, condition: str, budget_remaining: int) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(obj, dict):
        return False, ["stop check is not a JSON object"]
    decision = obj.get("decision")
    if decision not in DECISION_ENUM:
        errors.append(f"decision must be one of {sorted(DECISION_ENUM)}, got {decision!r}")
        return (False, errors)
    if decision == "STOP":
        cands = obj.get("final_candidates")
        if not isinstance(cands, list):
            errors.append("decision=STOP without final_candidates")
        else:
            if len(cands) != 3:
                errors.append(f"expected 3 final_candidates, got {len(cands)}")
            ranks = []
            for i, c in enumerate(cands):
                if not isinstance(c, dict):
                    errors.append(f"final_candidates[{i}] is not an object"); continue
                if not _tok(c.get("mutation")):
                    errors.append(f"final_candidates[{i}].mutation invalid")
                rk = c.get("rank")
                if isinstance(rk, bool) or not isinstance(rk, (int, float)) or int(rk) != rk:
                    errors.append(f"final_candidates[{i}].rank invalid: {rk!r}")
                else:
                    ranks.append(int(rk))
            if ranks and sorted(ranks) != [1, 2, 3]:
                errors.append(f"ranks must be exactly 1..3, got {sorted(ranks)}")
    return (len(errors) == 0, errors)
