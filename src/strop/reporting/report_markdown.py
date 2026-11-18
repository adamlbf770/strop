"""Rendering the report dictionary as a Markdown document."""

from typing import Any

from strop.formatting import fmt


def render_markdown(report: dict[str, Any]) -> str:
    """A readable report. Everything here comes from the stored results; nothing is estimated."""
    sections = [
        _title(report),
        _results(report),
        _changes(report),
        _history(report),
        _runs(report),
        _errors(report),
        _usage(report),
        _limitations(report),
    ]
    return "\n\n".join(s for s in sections if s) + "\n"


def _title(report: dict[str, Any]) -> str:
    project, prompt = report["project"], report["prompt"]
    lines = [
        f"# Strop report: {project['name']}",
        "",
        f"Task: {project['task']}",
        "",
        f"Selected prompt: v{prompt['selected']} (initial v{prompt['initial']}, "
        f"best v{prompt['best']}). Generated {report['generated_at']} by Strop "
        f"{report['strop_version']}.",
    ]
    models = report["models"]
    lines.append("")
    lines.append(
        "Models: " + ", ".join(f"{role} `{name}`" for role, name in models.items() if name)
    )
    return "\n".join(lines)


def _results(report: dict[str, Any]) -> str:
    primary = report["metrics"]["primary"]
    rows = []
    for label, key in (
        ("Initial prompt", "initial_validation"),
        ("Selected prompt", "selected_validation"),
        ("Initial prompt", "initial_test"),
        ("Selected prompt", "selected_test"),
    ):
        record = report["results"][key]
        if record is None:
            continue
        value = record["metrics"][primary]["value"]
        rows.append(
            f"| {label} (v{record['version'].lstrip('v')}) | {record['partition']} | "
            f"{fmt(value)} | {record['samples']} |"
        )
    if not rows:
        return "## Results\n\nNo evaluations are stored yet."
    header = f"| Prompt | Partition | {primary} | Examples |\n| --- | --- | --- | --- |"
    return "## Results\n\n" + header + "\n" + "\n".join(rows)


def _changes(report: dict[str, Any]) -> str:
    diff = report["diff"]
    if not diff["changed"]:
        return "## Prompt changes\n\nThe selected prompt is the initial prompt."
    return "## Prompt changes\n\n```diff\n" + _diff_lines(diff) + "\n```"


def _diff_lines(diff: dict[str, Any]) -> str:
    out = []
    for message in diff["messages"]:
        out.append(f"# message {message['index']} ({message['role']}): {message['status']}")
        for line in message["lines"]:
            prefix = {"add": "+", "remove": "-", "context": " ", "hunk": ""}[line["kind"]]
            out.append(f"{prefix}{line['text']}")
    return "\n".join(out)


def _history(report: dict[str, Any]) -> str:
    lines = [
        "## Version history",
        "",
        "| Version | Parent | Source | Decision | Change | Summary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for v in report["versions"]:
        delta = "" if v["primary_delta"] is None else f"{v['primary_delta']:+.4f}"
        parent = f"v{v['parent']}" if v["parent"] else ""
        lines.append(
            f"| v{v['version']} | {parent} | {v['source']} | {v['status'] or ''} | {delta} | "
            f"{_cell(v['change_summary'])} |"
        )
    return "\n".join(lines)


def _runs(report: dict[str, Any]) -> str:
    if not report["runs"]:
        return ""
    lines = [
        "## Optimization runs",
        "",
        "| Run | Status | Stopped because | Turns | Model calls |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['id']} | {r['status']} | {r['stop_reason'] or ''} | {r['turns']} | "
        f"{r['model_calls']} |"
        for r in report["runs"]
    ]
    return "\n".join(lines)


def _errors(report: dict[str, Any]) -> str:
    analysis = report["error_analysis"]
    if analysis is None:
        return ""
    lines = [
        "## Errors on validation",
        "",
        f"{analysis['failed']} of {analysis['examples']} examples failed (counts only).",
    ]
    lines += [f"- {code}: {count}" for code, count in sorted(analysis["codes"].items())]
    per_class = report["per_class"]
    if per_class:
        lines += [
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {name} | {fmt(s['precision'])} | {fmt(s['recall'])} | {fmt(s['f1'])} | "
            f"{s['support']} |"
            for name, s in per_class.items()
        ]
    return "\n".join(lines)


def _usage(report: dict[str, Any]) -> str:
    usage = report["usage"]
    cost = "unavailable" if usage["cost_usd"] is None else f"${usage['cost_usd']:.4f}"
    tokens = "unavailable" if usage["total_tokens"] is None else f"{usage['total_tokens']:,}"
    return (
        "## Usage\n\n"
        f"{usage['evaluations']} evaluations, {usage['model_calls']} model calls, "
        f"tokens {tokens}, cost {cost}."
    )


def _limitations(report: dict[str, Any]) -> str:
    return "## Limitations\n\n" + "\n".join(f"- {note}" for note in report["limitations"])


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")
