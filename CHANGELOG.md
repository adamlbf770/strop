# Changelog

All notable changes to Strop are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). While Strop is in alpha, minor versions may change
commands or the `strop.yaml` format.

## [0.1.0] - 2026-11-18

First version.

### Added

- `strop init`, `strop data check`, and `strop data split` to set up a project and create a
  reproducible, optionally stratified train, validation, and test split.
- Prompt files with chat messages, `{{variables}}`, and `<strop:fixed>` spans. Immutable prompt
  versions with `strop prompt add`, `strop prompt show`, `strop diff`, `strop history`, and
  `strop rollback`.
- Metrics: classification scores, exact match, JSON validity and field-level scores, string
  similarity, numeric closeness, rule checks, custom Python metrics, and LLM judge rubrics.
- `strop baseline`, `strop eval`, `strop analyze`, and `strop status`.
- `strop optimize`: a resumable loop that analyzes train failures, proposes revised prompts,
  scores them on validation, and applies acceptance rules (minimum gain, constraints, valid
  output rate, per-class regression, paired comparison). Manual approval with `strop approve`
  and `strop reject`, budgets, and a colored diff between the initial and final prompt.
- `strop test` with a locked test partition, `strop compare` for several target models,
  `strop export` for prompts, JSON and Markdown reports, and per-example results, and
  `strop clean`.
- Any provider and model supported by LiteLLM, with retries, a response cache, and `.env` support.
