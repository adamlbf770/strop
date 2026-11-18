"""Terminal rendering and JSON payloads for the test and compare commands."""

import math
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from strop.evaluation.models import EvaluationRecord
from strop.formatting import fmt
from strop.reporting.evaluation_view import record_payload, usage_line
from strop.services.compare import Comparison
from strop.services.final_test import FinalTest


def render_final_test(console: Console, result: FinalTest) -> None:
    """Summarize the held-out result next to the validation result it should be read against."""
    record = result.record
    primary = record.report.primary
    console.print(Text("Held-out test result", style="bold"), soft_wrap=True)
    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in ("Prompt", "Partition", primary, "Examples"):
        table.add_column(column)
    if result.baseline is not None:
        initial, base = result.baseline
        table.add_row(initial.ref + " (initial)", "test", _value(base), str(base.sample_count))
    if result.validation is not None:
        table.add_row(
            result.version.ref,
            "validation",
            _value(result.validation),
            str(result.validation.sample_count),
        )
    table.add_row(result.version.ref, "test", _value(record), str(record.sample_count))
    console.print(table)
    console.print(Text(usage_line(record.usage), style="dim"), soft_wrap=True)
    note = standard_error_note(record)
    if note:
        console.print(note, soft_wrap=True, highlight=False)
    console.print(
        "Do not change the prompt based on this result: the test set is now used up.",
        soft_wrap=True,
        highlight=False,
    )


def standard_error_note(record: EvaluationRecord) -> str | None:
    """A rough noise estimate for rate-like metrics, so small test sets are not over-read."""
    value = record.report.primary_value.value
    n = record.report.primary_value.samples
    if value is None or n == 0 or not 0.0 <= value <= 1.0:
        return None
    error = math.sqrt(value * (1 - value) / n)
    return (
        f"Note: with {n} test examples, differences smaller than about {error:.2f} "
        f"(one standard error for a rate near {value:.2f}) may be noise."
    )


def render_comparison(console: Console, comparison: Comparison) -> None:
    title = f"{comparison.version.ref} on {comparison.partition.value}, same rows for every model"
    console.print(Text(title, style="bold"), soft_wrap=True)
    table = Table(box=None, pad_edge=False, header_style="bold")
    primary = comparison.results[0][1].report.primary
    for column in ("Model", primary, "Valid output rate", "Errors", "Cost", "Mean latency"):
        table.add_column(column)
    for name, record in comparison.results:
        validity = record.report.values.get("valid_output_rate")
        cost = "unavailable" if record.usage.cost_usd is None else f"${record.usage.cost_usd:.4f}"
        latency = (
            fmt(record.usage.mean_latency_s, 2) + "s" if record.usage.mean_latency_s else "n/a"
        )
        table.add_row(
            name,
            _value(record),
            fmt(validity.value if validity else None),
            f"{record.error_count}/{record.sample_count}",
            cost,
            latency,
        )
    console.print(table)


def _value(record: EvaluationRecord) -> str:
    return fmt(record.report.primary_value.value)


def final_test_payload(result: FinalTest) -> dict[str, Any]:
    ref = result.version.ref
    return {
        "version": ref,
        "test": record_payload(result.record, ref),
        "validation": record_payload(result.validation, ref) if result.validation else None,
        "initial": (
            record_payload(result.baseline[1], result.baseline[0].ref) if result.baseline else None
        ),
    }


def comparison_payload(comparison: Comparison) -> dict[str, Any]:
    ref = comparison.version.ref
    return {
        "version": ref,
        "partition": comparison.partition.value,
        "models": [
            {"model": name, **record_payload(record, ref)} for name, record in comparison.results
        ],
    }
