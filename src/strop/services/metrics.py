"""Building the metric suite a project asks for."""

from strop.config import StropConfig
from strop.config.enums import ModelRole
from strop.config.llm import ModelSpec
from strop.errors import ConfigError
from strop.llm.client import LLMClient
from strop.metrics.judge import JudgeMetric
from strop.metrics.python_metric import METRIC_NAME, PythonMetric
from strop.metrics.registry import MetricContext, build_metric
from strop.metrics.suite import MetricSuite
from strop.metrics.types import Direction, Metric
from strop.services.project import Project

JudgeAccess = tuple[LLMClient, ModelSpec]


def build_suite(project: Project, judge: JudgeAccess | None = None) -> MetricSuite:
    """The primary, secondary, rule, custom, judge, and constraint metrics from strop.yaml."""
    config = project.config
    extra: dict[str, Metric] = {}
    if config.metrics.python is not None:
        python = config.metrics.python
        extra[METRIC_NAME] = PythonMetric(
            project.root / python.file,
            timeout_s=python.timeout_s,
            direction=Direction(python.direction),
        )
    for name, spec in config.metrics.judges.items():
        if judge is None:
            raise ConfigError(
                f"judge metric '{name}' needs a judge model; set models.judge.model in "
                "strop.yaml (strop init --judge-model writes it)"
            )
        client, model = judge
        extra[f"judge:{name}"] = JudgeMetric(
            name,
            spec,
            client,
            model,
            config.project.task_description,
            config.execution.max_concurrency,
        )
    validate_metric_names(config)
    context = MetricContext(labels=config.output.labels, rules=config.metrics.rules)
    suite = MetricSuite.from_config(config.metrics, context, extra)
    _check_references(project, suite)
    return suite


def validate_metric_names(config: StropConfig) -> None:
    """Fail early, with the list of valid names, if a configured metric does not exist."""
    metrics = config.metrics
    context = MetricContext(labels=config.output.labels, rules=metrics.rules)
    for name in (metrics.primary, *metrics.secondary, *metrics.constraints):
        if name == METRIC_NAME:
            if metrics.python is None:
                raise ConfigError(
                    "metric 'custom' needs metrics.python in strop.yaml "
                    "(strop init --metric-file writes it)"
                )
        elif name.startswith("judge:"):
            if name.removeprefix("judge:") not in metrics.judges:
                raise ConfigError(
                    f"metric '{name}' has no rubric under metrics.judges in strop.yaml "
                    "(strop init --rubric-file writes it)"
                )
        else:
            build_metric(name, context)


def judge_model(project: Project) -> ModelSpec | None:
    """The judge model if any judge rubric is configured."""
    if not project.config.metrics.judges:
        return None
    return project.config.models.resolve(ModelRole.JUDGE)


def _check_references(project: Project, suite: MetricSuite) -> None:
    """Metrics that compare with expected outputs need an expected column."""
    if project.config.dataset.expected_column is not None:
        return
    config = project.config.metrics
    for name in (config.primary, *config.constraints):
        if suite.metric(name).needs_expected:
            raise ConfigError(
                f"metric '{name}' compares outputs with expected values, but no dataset column "
                "has the role 'expected'; set --expected or choose another metric"
            )
