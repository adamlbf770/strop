import json
from pathlib import Path

from tests.helpers import CliRunner

EDITED = """\
messages:
  - role: system
    content: |
      Classify a customer message.
      Respond with exactly one of these labels and nothing else: billing, refund.
      A request for money back is "refund".
  - role: user
    content: |
      message: {{message}}
optimize:
  - system
"""


def edited_file(project: Path, name: str = "edited.yaml", text: str = EDITED) -> Path:
    path = project / name
    path.write_text(text, encoding="utf-8")
    return path


def add_edit(run_cli: CliRunner, project: Path, *extra: str) -> tuple[int, str, str]:
    edited_file(project)
    return run_cli(
        "prompt", "add", "--dir", str(project), "--file", str(project / "edited.yaml"), *extra
    )


def test_history_is_empty_before_the_first_run(run_cli: CliRunner, project: Path) -> None:
    code, out, _ = run_cli("history", "--dir", str(project))
    assert code == 0
    assert "No prompt versions yet" in out


def test_prompt_add_registers_the_initial_prompt_first(run_cli: CliRunner, project: Path) -> None:
    code, out, _ = add_edit(run_cli, project, "--reason", "clarify refunds")
    assert code == 0
    assert "Saved v2 (parent v1)" in out
    _, history, _ = run_cli("history", "--dir", str(project))
    lines = history.splitlines()
    assert lines[0].startswith("* v1")
    assert lines[1].startswith("  v2")
    assert "manual edit" in lines[1]


def test_use_makes_the_new_version_best(run_cli: CliRunner, project: Path) -> None:
    code, out, _ = add_edit(run_cli, project, "--use")
    assert code == 0
    assert "v2 is now the best prompt" in out
    _, history, _ = run_cli("history", "--dir", str(project))
    assert history.splitlines()[1].startswith("* v2")


def test_identical_prompt_is_rejected(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project)
    code, _, err = add_edit(run_cli, project)
    assert code == 2
    assert "identical to the existing version v2" in err


def test_unknown_variable_is_rejected(run_cli: CliRunner, project: Path) -> None:
    path = edited_file(project, text=EDITED.replace("{{message}}", "{{msg}}"))
    code, _, err = run_cli("prompt", "add", "--dir", str(project), "--file", str(path))
    assert code == 2
    assert "msg" in err
    assert "input columns: message" in err


def test_show_prints_roles_and_editability(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project)
    _, out, _ = run_cli("prompt", "show", "v2", "--dir", str(project))
    assert out.startswith("v2 (parent v1, user)")
    assert "--- system (editable)" in out
    assert "--- user (frozen)" in out
    assert 'A request for money back is "refund".' in out


def test_show_defaults_to_the_best_version(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project)
    _, out, _ = run_cli("prompt", "show", "--dir", str(project), "--json")
    payload = json.loads(out)
    assert payload["version"] == 1
    assert payload["best"] is True


def test_show_reports_an_unknown_version(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project)
    code, _, err = run_cli("prompt", "show", "v9", "--dir", str(project))
    assert code == 2
    assert "v9 does not exist" in err


def test_diff_plain_output_is_a_unified_diff(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project, "--use")
    code, out, _ = run_cli("diff", "--dir", str(project), "--color", "never")
    assert code == 0
    assert "v1 -> v2" in out
    assert '+A request for money back is "refund".' in out
    assert "== message 2 (user): frozen" in out
    assert "Summary: +1 line(s), -0 line(s)" in out
    assert "\x1b[" not in out


def test_diff_color_always_emits_ansi_styles(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project, "--use")
    _, out, _ = run_cli("diff", "--dir", str(project), "--color", "always")
    assert "\x1b[" in out


def test_diff_of_one_version_compares_with_its_parent(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project)
    _, out, _ = run_cli("diff", "v2", "--dir", str(project), "--color", "never")
    assert "v1 -> v2" in out


def test_diff_json_output(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project)
    _, out, _ = run_cli("diff", "v1", "v2", "--dir", str(project), "--json")
    payload = json.loads(out)
    assert (payload["from"], payload["to"], payload["changed"]) == ("v1", "v2", True)
    assert payload["stats"]["added_lines"] == 1


def test_diff_without_versions_explains_what_to_do(run_cli: CliRunner, project: Path) -> None:
    code, _, err = run_cli("diff", "--dir", str(project))
    assert code == 2
    assert "strop baseline" in err


def test_rollback_moves_only_the_best_pointer(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project, "--use")
    code, out, _ = run_cli("rollback", "v1", "--dir", str(project))
    assert code == 0
    assert "Best prompt is now v1 (was v2). No versions were deleted." in out
    _, history, _ = run_cli("history", "--dir", str(project))
    assert "v2" in history
    assert history.splitlines()[0].startswith("* v1")


def test_history_json_output(run_cli: CliRunner, project: Path) -> None:
    add_edit(run_cli, project)
    _, out, _ = run_cli("history", "--dir", str(project), "--json")
    payload = json.loads(out)
    assert payload["best"] == 1
    assert [v["version"] for v in payload["versions"]] == [1, 2]
