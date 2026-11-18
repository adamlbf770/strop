"""Deleting stored data on request."""

import shutil

from strop.errors import ConfigError
from strop.services.project import Project
from strop.storage.cache import ResponseCache

CHOICES = "--cache, --outputs, or --all"


def clean_project(
    project: Project, *, cache: bool, outputs: bool, everything: bool, confirmed: bool
) -> list[str]:
    """Delete the chosen data and report what was removed; nothing goes without ``confirmed``."""
    if not (cache or outputs or everything):
        raise ConfigError(f"choose what to delete: {CHOICES}")
    if not confirmed:
        raise ConfigError("this permanently deletes stored data; pass --yes to confirm")
    messages = []
    if everything:
        shutil.rmtree(project.paths.state_dir, ignore_errors=True)
        return [
            f"Deleted {project.paths.state_dir.name}/ (database, cache, prompt versions, results)",
            "strop.yaml, the prompt file, and the dataset were not touched.",
        ]
    with project.database() as conn:
        if cache or outputs:
            removed = ResponseCache(conn).clear()
            messages.append(f"Deleted {removed} cached model response(s).")
        if outputs:
            cleared = conn.execute(
                "UPDATE example_results SET raw_output = '', value = NULL, parsed_json = NULL, "
                "detail = NULL, feedback = NULL"
            ).rowcount
            conn.execute("UPDATE candidates SET raw_response = ''")
            messages.append(f"Cleared the stored model output of {cleared} example result(s).")
    return messages
