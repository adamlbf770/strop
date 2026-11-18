"""Evaluating the full set of metrics a project asks for."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from strop.config.metrics import Bound, MetricsConfig
from strop.metrics.registry import MetricContext, build_metric
from strop.metrics.types import Direction, Example, Metric, MetricValue

ALWAYS_REPORTED = "valid_output_rate"


@dataclass(frozen=True)
class MetricReport:
    """Every metric value computed for one set of examples."""

    primary: str
    values: dict[str, MetricValue]

    @property
    def primary_value(self) -> MetricValue:
        return self.values[self.primary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary,
            "values": {
                name: {
                    "value": m.value,
                    "samples": m.samples,
                    "subjective": m.subjective,
                    "details": m.details,
                }
                for name, m in self.values.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricReport":
        values = {
            name: MetricValue(
                name, item["value"], item["samples"], item["subjective"], item["details"]
            )
            for name, item in data["values"].items()
        }
        return cls(data["primary"], values)

    def violations(self, bounds: Mapping[str, Bound]) -> list[str]:
        """Constraint violations as readable sentences; empty when all constraints hold."""
        problems = []
        for name, bound in bounds.items():
            metric = self.values.get(name)
            if metric is None or metric.value is None:
                problems.append(f"{name} could not be computed")
            elif bound.min is not None and metric.value < bound.min:
                problems.append(f"{name} is {metric.value:.4f}, below the minimum {bound.min}")
            elif bound.max is not None and metric.value > bound.max:
                problems.append(f"{name} is {metric.value:.4f}, above the maximum {bound.max}")
        return problems


class MetricSuite:
    """The primary metric, secondary metrics, rule checks, and constraint metrics together."""

    def __init__(self, primary: str, metrics: Mapping[str, Metric]) -> None:
        self.primary = primary
        self._metrics = dict(metrics)

    @classmethod
    def from_config(
        cls,
        config: MetricsConfig,
        context: MetricContext,
        extra: Mapping[str, Metric] | None = None,
    ) -> "MetricSuite":
        extra = extra or {}
        names = [
            config.primary,
            *config.secondary,
            *config.constraints,
            *(f"rule:{name}" for name in config.rules),
            *extra,
            ALWAYS_REPORTED,
        ]
        metrics = {
            name: extra[name] if name in extra else build_metric(name, context)
            for name in dict.fromkeys(names)
        }
        return cls(config.primary, metrics)

    @property
    def names(self) -> list[str]:
        return list(self._metrics)

    def metric(self, name: str) -> Metric:
        return self._metrics[name]

    def direction(self, name: str) -> Direction:
        return self._metrics[name].direction

    async def prepare(self, examples: Sequence[Example]) -> None:
        """Run every metric's batched or asynchronous setup before scoring."""
        for metric in self._metrics.values():
            await metric.prepare(examples)

    def evaluate(self, examples: Sequence[Example]) -> MetricReport:
        values = {name: metric.compute(examples) for name, metric in self._metrics.items()}
        return MetricReport(self.primary, values)
