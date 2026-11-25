"""Prompts for the LLM agents. Every agent output is re-validated against the notebook."""
from __future__ import annotations

import json
from typing import Any, Dict, List

CLAIM_SYSTEM = """You extract atomic factual claims from a derived artifact (slides, paper, summary) about a computational notebook.
Rules:
- One claim = one checkable statement. Split compound sentences ("accuracy was 85% and F1 was 0.80" -> two claims).
- Make each claim self-contained: replace pronouns and vague references ("it", "this model", "the dataset") using the
  unit's context heading or earlier units. Never add facts, numbers or qualifiers that are not in the source.
- Copy every number exactly as written in the source.
- Skip headings, questions, agenda items, and non-factual filler.
- category: "numeric" (exact numbers, percentages, sample sizes, metrics), "trend" (direction of change),
  "methodological" (data preparation, splits, baselines, procedures), "qualitative" (interpretive conclusions without numbers).
Output: {"claims": [{"unit_id": <int>, "claim": "<self-contained claim>", "category": "<category>"}]}"""


def claim_user_prompt(units: List[Dict[str, Any]]) -> str:
    return "Units (in document order):\n" + json.dumps(units, indent=1, ensure_ascii=False)


PLANNER_SYSTEM = """You are the PLANNER agent in a notebook audit. For each claim, decide which notebook cells contain
the evidence needed to verify it: cells whose printed outputs show the claimed numbers/metrics, cells whose code implements
the claimed procedure (splits, scaling, baselines), and cells that plot the claimed trend.
Only use cell indices that appear in the notebook index. Prefer 1-4 cells per claim. If nothing is relevant, return [].
Output: {"plans": [{"claim_id": "C1", "target_cells": [4, 7], "rationale": "<short reason>"}]}"""


def planner_user_prompt(notebook_index: str, claims: List[Dict[str, Any]]) -> str:
    return f"NOTEBOOK INDEX:\n{notebook_index}\n\nCLAIMS:\n{json.dumps(claims, indent=1, ensure_ascii=False)}"


EXECUTOR_SYSTEM = """You are the EXECUTOR agent. Given one claim and the full code + outputs of a few notebook cells,
extract the exact evidence values relevant to the claim: printed numbers, metric values, matrix/array dimensions, sample
counts, and code parameters (e.g. test_size, stratify, random_state, n_splits).
Rules:
- "quote" MUST be copied verbatim (character for character) from the cell's code or output. Do not paraphrase.
- "value" is the number inside the quote as a plain decimal (no % sign, no commas); use null for non-numeric facts.
- "label" names what the value measures (e.g. "test accuracy", "number of documents", "test_size").
- Do not judge the claim; only extract.
Output: {"values": [{"cell_index": 4, "location": "output"|"code", "quote": "...", "value": 420, "label": "..."}]}"""


def executor_user_prompt(claim: Dict[str, Any], cells_text: str) -> str:
    return f"CLAIM:\n{json.dumps(claim, ensure_ascii=False)}\n\nCELLS:\n{cells_text}"


VERIFIER_SYSTEM = """You are the VERIFIER agent in a strict notebook audit. Decide whether the claim is backed by the
execution evidence. Apply these rules strictly:
1. SUPPORTED only if a cited cell output or code line directly shows the claimed fact. Rounding of the last stated digit
   is allowed; changes in magnitude, unit, split (train vs test) or metric (accuracy vs F1) are not.
2. CONTRADICTED if a cited output shows a different value for the same quantity, an opposite trend, or code that does
   something different from the claimed procedure.
3. METHODOLOGICAL_RISK if the claim matches the outputs but the flagged risks (leakage, missing baseline, borderline
   threshold) undermine what the claim asserts.
4. UNVERIFIABLE if the evidence is absent, ambiguous, or the claim is an interpretation that outputs cannot establish.
   General plausibility or "consistent with" is NOT support.
5. Every SUPPORTED / CONTRADICTED / METHODOLOGICAL_RISK verdict must cite at least one verbatim quote from a cell.
Output: {"verdict": "...", "confidence": 0.0-1.0, "rationale": "<1-3 sentences>",
         "citations": [{"cell_index": 4, "quote": "<verbatim text from that cell>"}]}"""


def verifier_user_prompt(claim: Dict[str, Any], rule_result: Dict[str, Any], cells_text: str, risks: List[str]) -> str:
    return (
        f"CLAIM:\n{json.dumps(claim, ensure_ascii=False)}\n\n"
        f"DETERMINISTIC RULE CHECKS (authoritative for exact numbers):\n{json.dumps(rule_result, ensure_ascii=False)}\n\n"
        f"FLAGGED METHODOLOGICAL RISKS:\n" + ("\n".join(f"- {r}" for r in risks) or "- none") + "\n\n"
        f"EVIDENCE CELLS:\n{cells_text}"
    )
