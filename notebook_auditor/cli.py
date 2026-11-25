"""Command-line interface: `notebook-auditor run|parse|claims`."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from . import __version__
from .claim_extractor import extract_claims
from .llm import make_client
from .models import VERDICTS
from .parser import load_notebook
from .pipeline import AuditPipeline
from .report import VERDICT_ICON, write_reports


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="notebook-auditor", description="Audit Jupyter notebooks and their derived artifacts for evidence traceability and methodological risk.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full audit and write audit_report.md / audit_report.json")
    run.add_argument("--notebook", "-n", required=True, help="Path to the .ipynb file")
    run.add_argument("--slides", "-s", action="append", default=[], help="Derived artifact (markdown/text). Repeatable.")
    run.add_argument("--output-dir", "-o", default="./audit_reports", help="Directory for the reports (default: ./audit_reports)")
    run.add_argument("--provider", default="auto", choices=["auto", "anthropic", "openai", "litellm", "none"],
                     help="LLM provider for the agents. 'auto' uses ANTHROPIC_API_KEY / OPENAI_API_KEY if set, else heuristic-only.")
    run.add_argument("--model", default=None, help="Model name for the provider (default per provider)")
    run.add_argument("--include-notebook-markdown", action="store_true",
                     help="Also extract claims from the notebook's markdown cells (default when no --slides given)")
    run.add_argument("--threshold-tolerance", type=float, default=0.005, help="Borderline window around α / metric thresholds (default 0.005)")
    run.add_argument("--format", choices=["md", "json", "both"], default="both")
    run.add_argument("--fail-on", choices=["none", "contradicted", "risk", "any"], default="none",
                     help="Exit non-zero if claims are CONTRADICTED / have METHODOLOGICAL_RISK / either (for CI)")
    run.add_argument("--quiet", "-q", action="store_true")

    parse = sub.add_parser("parse", help="Print the linear text view of a notebook ([Cell N - Code] / [Cell N - Output])")
    parse.add_argument("--notebook", "-n", required=True)
    parse.add_argument("--max-output-chars", type=int, default=4000)

    claims = sub.add_parser("claims", help="Extract and print claims from artifacts (no audit)")
    claims.add_argument("--slides", "-s", action="append", default=[], required=True)
    claims.add_argument("--provider", default="none", choices=["auto", "anthropic", "openai", "litellm", "none"])
    claims.add_argument("--model", default=None)
    return p


def _cmd_run(args: argparse.Namespace) -> int:
    nb = load_notebook(args.notebook)
    llm = make_client(args.provider, args.model)
    include_md = args.include_notebook_markdown or not args.slides
    claims, warnings = extract_claims(args.slides, nb, include_md, llm=llm)
    if not args.quiet:
        mode = f"LLM agents via {llm.label}" if llm else "heuristic agents (no LLM)"
        print(f"Auditing {args.notebook} · {len(nb.cells)} cells · {len(claims)} claims · {mode}", file=sys.stderr)
    result = AuditPipeline(llm, args.threshold_tolerance).run(nb, claims, args.slides, warnings)
    formats = ("md", "json") if args.format == "both" else (args.format,)
    paths = write_reports(result, args.output_dir, nb, formats)
    if llm is not None and llm.calls and not llm.successes:
        print(f"warning: every {llm.label} call failed ({llm.errors[0] if llm.errors else 'unknown error'}); "
              "this report is heuristic-only.", file=sys.stderr)
    if not args.quiet:
        print(f"  mode: {result.mode}")
        s = result.summary()
        print("  " + "  ".join(f"{VERDICT_ICON[v]} {v}: {s['verdicts'][v]}" for v in VERDICTS))
        print("  findings: " + ", ".join(f"{k}={v}" for k, v in s["risks_by_severity"].items()))
        for kind, path in paths.items():
            print(f"  wrote {path}")
    counts = result.summary()["verdicts"]
    if args.fail_on == "contradicted" and counts["CONTRADICTED"]:
        return 1
    if args.fail_on == "risk" and counts["METHODOLOGICAL_RISK"]:
        return 1
    if args.fail_on == "any" and (counts["CONTRADICTED"] or counts["METHODOLOGICAL_RISK"]):
        return 1
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    nb = load_notebook(args.notebook)
    print(nb.to_linear_text(max_output_chars=args.max_output_chars))
    return 0


def _cmd_claims(args: argparse.Namespace) -> int:
    llm = make_client(args.provider, args.model)
    claims, warnings = extract_claims(args.slides, None, False, llm=llm)
    print(json.dumps([{"id": c.claim_id, "category": c.category, "claim": c.text, "source": c.source, "tags": c.tags} for c in claims], indent=2, ensure_ascii=False))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        return {"run": _cmd_run, "parse": _cmd_parse, "claims": _cmd_claims}[args.command](args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
