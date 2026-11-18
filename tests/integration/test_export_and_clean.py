import csv
import json
import sqlite3
from pathlib import Path

from strop.prompts import parse_prompt
from tests.helpers import CliRunner
from tests.integration.conftest import Install


def finished(run_cli: CliRunner, project: Path, *, tested: bool = False) -> None:
    assert run_cli("optimize", "--dir", str(project), "--candidates", "2")[0] == 0
    if tested:
        assert run_cli("test", "--dir", str(project), "--confirm")[0] == 0


def test_export_writes_the_prompt_report_and_results(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready, tested=True)
    code, out, _ = run_cli("export", "--dir", str(ready))
    assert code == 0
    assert "Exported v2:" in out
    assert "results.csv contains model outputs" in out
    exports = ready / "exports"
    assert {p.name for p in exports.iterdir()} == {
        "prompt_v2.md",
        "report.json",
        "report.md",
        "results.csv",
    }
    report = json.loads((exports / "report.json").read_text(encoding="utf-8"))
    assert report["prompt"] == {"selected": 2, "initial": 1, "best": 2}
    assert report["results"]["selected_validation"]["metrics"]["macro_f1"]["value"] == 1.0
    assert report["results"]["selected_test"]["partition"] == "test"
    assert report["results"]["initial_validation"]["metrics"]["macro_f1"]["value"] < 1.0
    assert [v["version"] for v in report["versions"]] == [1, 2, 3]
    assert report["diff"]["changed"] is True
    assert report["dataset"]["rows"] == 100
    assert report["usage"]["cost_usd"] is None
    assert "Cost is unavailable" in " ".join(report["limitations"])
    assert "has not been evaluated" not in " ".join(report["limitations"])


def test_exported_prompt_matches_the_stored_version_exactly(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready)
    _, shown, _ = run_cli("prompt", "show", "v2", "--dir", str(ready), "--json")
    stored = [m["content"] for m in json.loads(shown)["messages"]]
    run_cli("export", "--dir", str(ready), "--prompt-format", "yaml")
    exported = parse_prompt((ready / "exports" / "prompt_v2.yaml").read_text("utf-8"), ".yaml")
    assert [m.content for m in exported.messages] == stored
    markdown = run_cli("export", "--dir", str(ready), "--force")[0]
    assert markdown == 0
    assert all(
        content in (ready / "exports" / "prompt_v2.md").read_text("utf-8") for content in stored
    )


def test_a_written_prompt_file_is_recognized_as_an_existing_version(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready)
    code, _, _ = run_cli("export", "--dir", str(ready), "--write-prompt", "best.yaml")
    assert code == 0
    code, _, err = run_cli("prompt", "add", "--dir", str(ready), "--file", str(ready / "best.yaml"))
    assert code == 2
    assert "identical to the existing version v2" in err


def test_export_does_not_overwrite_without_force(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready)
    assert run_cli("export", "--dir", str(ready))[0] == 0
    code, _, err = run_cli("export", "--dir", str(ready))
    assert code == 2
    assert "--force" in err
    assert run_cli("export", "--dir", str(ready), "--force")[0] == 0


def test_results_csv_lists_every_example_and_can_be_skipped(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready, tested=True)
    run_cli("export", "--dir", str(ready))
    with (ready / "exports" / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["partition"] for r in rows} == {"validation", "test"}
    assert len(rows) == 40
    assert rows[0]["model_output"] in {"refund", "billing"}
    code, out, _ = run_cli("export", "--dir", str(ready), "--out", "slim", "--no-results")
    assert code == 0
    assert not (ready / "slim" / "results.csv").exists()
    assert "results.csv contains" not in out


def test_report_says_when_the_test_set_has_not_been_used(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready)
    run_cli("export", "--dir", str(ready))
    text = (ready / "exports" / "report.md").read_text(encoding="utf-8")
    assert "not been evaluated on the test partition yet" in text
    assert text.startswith("# Strop report: demo")
    assert "```diff" in text
    assert "## Version history" in text
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_export_json_output(run_cli: CliRunner, ready: Path, install_models: Install) -> None:
    install_models()
    _, out, _ = run_cli("export", "--dir", str(ready), "--json")
    payload = json.loads(out)
    assert payload["version"] == 1
    assert len(payload["files"]) == 4


def test_clean_requires_a_choice_and_confirmation(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    code, _, err = run_cli("clean", "--dir", str(ready))
    assert code == 2
    assert "--cache, --outputs, or --all" in err
    code, _, err = run_cli("clean", "--dir", str(ready), "--cache")
    assert code == 2
    assert "--yes" in err


def test_clean_cache_and_outputs_keep_scores_and_versions(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready)
    code, out, _ = run_cli("clean", "--dir", str(ready), "--outputs", "--yes")
    assert code == 0
    assert "Cleared the stored model output" in out
    conn = sqlite3.connect(ready / ".strop" / "strop.db")
    assert (
        conn.execute("SELECT COUNT(*) FROM example_results WHERE raw_output != ''").fetchone()[0]
        == 0
    )
    assert conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0] == 3
    _, status, _ = run_cli("status", "--dir", str(ready))
    assert "macro_f1=1.0000" in status


def test_clean_all_removes_the_state_directory_only(
    run_cli: CliRunner, ready: Path, install_models: Install
) -> None:
    install_models()
    finished(run_cli, ready)
    code, out, _ = run_cli("clean", "--dir", str(ready), "--all", "--yes")
    assert code == 0
    assert "were not touched" in out
    assert not (ready / ".strop").exists()
    assert (ready / "strop.yaml").exists()
    assert (ready / "prompt.yaml").exists()
    assert (ready / "data.csv").exists()
