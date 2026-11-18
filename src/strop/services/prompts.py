"""Prompt version workflows."""

from pathlib import Path

from strop.errors import ConfigError
from strop.prompts import PromptTemplate, check_variables, load_prompt_file
from strop.prompts.versions import PromptVersion, parse_version_ref
from strop.services.project import Project
from strop.storage.prompts import PromptRepository


def load_project_prompt(project: Project, path: Path | None = None) -> PromptTemplate:
    """Parse a prompt file and check it only uses variables the dataset provides."""
    template = load_prompt_file(path or project.root / project.config.prompt.file)
    check_variables(template.variables, project.config.dataset.input_columns)
    return template


def ensure_initial_version(project: Project, repo: PromptRepository) -> PromptVersion:
    """Register the prompt file as v1 on first use, and make it the best prompt."""
    existing = repo.first()
    if existing is not None:
        return existing
    template = load_project_prompt(project)
    version = repo.add(
        template,
        parent_number=None,
        change_summary="initial prompt",
        reason=f"registered from {project.config.prompt.file}",
    )
    repo.set_best(version.number)
    return version


def has_unregistered_edits(project: Project, repo: PromptRepository) -> bool:
    """True if the prompt file differs from every stored version."""
    template = load_prompt_file(project.root / project.config.prompt.file)
    return repo.find_by_hash(template.content_hash) is None


def add_version_from_file(
    project: Project,
    repo: PromptRepository,
    file: Path,
    *,
    parent: str | None,
    reason: str,
    use: bool,
) -> PromptVersion:
    """Save a manual edit as a new immutable version, optionally making it the best prompt."""
    ensure_initial_version(project, repo)
    template = load_project_prompt(project, file)
    duplicate = repo.find_by_hash(template.content_hash)
    if duplicate is not None:
        raise ConfigError(f"this prompt is identical to the existing version {duplicate.ref}")
    parent_number = resolve_version(repo, parent).number
    version = repo.add(
        template,
        parent_number=parent_number,
        change_summary="manual edit",
        reason=reason,
        source="user",
    )
    if use:
        repo.set_best(version.number)
    return version


def resolve_version(repo: PromptRepository, ref: str | None) -> PromptVersion:
    """The version named by ``ref`` (like ``v3``), or the best version when ``ref`` is None."""
    if ref is not None:
        return repo.get(parse_version_ref(ref))
    best = repo.best()
    if best is None:
        raise ConfigError("no prompt versions yet; run 'strop baseline' or 'strop prompt add'")
    return best
