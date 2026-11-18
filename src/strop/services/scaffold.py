"""Creating a new Strop project on disk."""

from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from strop import __version__
from strop.config import OutputType, StropConfig, dump_config
from strop.errors import ConfigError
from strop.storage import ProjectPaths, open_database


def scaffold_project(config: StropConfig, root: Path, *, force: bool = False) -> list[str]:
    """Write strop.yaml, a starter prompt if none exists, and the state directory.

    Returns the entries that were created or rewritten, relative to ``root``.
    """
    dataset_file = root / config.dataset.path
    if not dataset_file.is_file():
        raise ConfigError(f"dataset file not found: {dataset_file}")

    paths = ProjectPaths(root)
    dump_config(config, paths.config_file, force=force)
    created = [paths.config_file.name]

    prompt_file = root / config.prompt.file
    if not prompt_file.exists():
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(starter_prompt(config), encoding="utf-8")
        created.append(str(config.prompt.file))

    conn = open_database(paths)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            [
                ("project_name", config.project.name),
                ("strop_version", __version__),
                ("created_at", datetime.now(UTC).isoformat()),
            ],
        )
    finally:
        conn.close()
    created.append(f"{paths.state_dir.name}/")
    return created


def data_recipients(config: StropConfig) -> list[str]:
    """Provider names that will receive dataset content, from the configured models."""
    models = config.models
    specs = (models.target, models.optimizer, models.analyzer, models.judge)
    return sorted({spec.model.split("/", 1)[0] for spec in specs if spec is not None})


def starter_prompt(config: StropConfig) -> str:
    """A minimal prompt file the user is expected to edit before running."""
    system = config.project.task_description.strip()
    if config.output.type is OutputType.LABEL:
        labels = ", ".join(config.output.labels)
        system += f"\nRespond with exactly one of these labels and nothing else: {labels}."
    elif config.output.type is OutputType.JSON:
        system += "\nRespond with a single valid JSON object and nothing else."
    user = "\n".join(f"{name}: {{{{{name}}}}}" for name in config.dataset.input_columns)
    document = {
        "messages": [
            {"role": "system", "content": LiteralScalarString(system + "\n")},
            {"role": "user", "content": LiteralScalarString(user + "\n")},
        ],
        "optimize": ["system"],
    }
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    buffer = StringIO()
    yaml.dump(document, buffer)
    return buffer.getvalue()
