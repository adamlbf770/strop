"""The ``strop analyze`` command."""

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer

from strop.cli.evaluate import NoCacheOption
from strop.cli.options import DirOption, JsonOption, emit_json
from strop.formatting import shorten
from strop.prompts.versions import PromptVersion
from strop.services.analysis import TurnAnalysis, run_analysis
from strop.services.project import open_project


def analyze_command(
    version: Annotated[
        str | None, typer.Argument(help="Version such as v2; default: best.")
    ] = None,
    patterns: Annotated[
        bool, typer.Option("--patterns", help="Ask the analyzer model for failure patterns.")
    ] = False,
    examples: Annotated[int | None, typer.Option(help="Failed examples to list.", min=1)] = None,
    no_cache: NoCacheOption = False,
    as_json: JsonOption = False,
    directory: DirOption = Path(),
) -> None:
    """Explain how a prompt fails on the training data.

    Counts and examples are observed facts. With --patterns, an LLM also proposes recurring
    failure patterns; those are hypotheses, not verified findings.
    """
    project = open_project(directory)
    with project.database() as conn:
        selected, analysis = asyncio.run(
            run_analysis(
                project,
                conn,
                version,
                patterns=patterns,
                examples=examples,
                use_cache=not no_cache,
            )
        )
    if as_json:
        emit_json(_payload(selected, analysis))
        return
    typer.echo(_report(selected, analysis, patterns))


def _payload(version: PromptVersion, analysis: TurnAnalysis) -> dict[str, Any]:
    return {
        "version": version.ref,
        "evaluation": analysis.train_record.id,
        **analysis.to_json(),
        "examples": [
            {"id": e.row_id, "expected": e.expected, "got": e.got, "problem": e.code}
            for e in analysis.evidence
        ],
    }


def _report(version: PromptVersion, analysis: TurnAnalysis, asked: bool) -> str:
    facts = analysis.facts
    lines = [
        f"Analysis of {version.ref} on train (evaluation {analysis.train_record.id}): "
        f"{facts.failed} of {facts.total} examples failed.",
        "Observed:",
    ]
    lines += [f"  {code:<18} {count}" for code, count in sorted(facts.codes.items())] or [
        "  no failures"
    ]
    if facts.confusions:
        lines.append("Most common confusions (expected -> got):")
        lines += [f"  {e} -> {g}: {n}" for e, g, n in facts.confusions]
    if analysis.evidence:
        lines.append(f"Failed examples (showing {len(analysis.evidence)} of {facts.failed}):")
        lines += [
            f"  {item.row_id}: {item.code}, expected {shorten(item.expected or '')!r}, "
            f"got {shorten(item.got)!r}"
            for item in analysis.evidence
        ]
    if asked:
        lines.append("Hypotheses (proposed by an LLM, not verified):")
        lines += [
            f"  {p.name}: {p.description} (examples: {', '.join(p.example_ids)})"
            for p in analysis.patterns
        ] or ["  none"]
    return "\n".join(lines)
