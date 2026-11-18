"""Building the metric suite a project asks for."""

from strop.metrics.registry import MetricContext
from strop.metrics.suite import MetricSuite
from strop.services.project import Project


def build_suite(project: Project) -> MetricSuite:
    """The primary, secondary, rule, and constraint metrics from strop.yaml."""
    config = project.config
    context = MetricContext(labels=config.output.labels, rules=config.metrics.rules)
    return MetricSuite.from_config(config.metrics, context)
