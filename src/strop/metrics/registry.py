"""Resolving metric names from strop.yaml to metric objects."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from strop.config.metrics import RuleSpec
from strop.errors import ConfigError
from strop.metrics.basic import ExactMatch, ValidOutputRate
from strop.metrics.classification import Accuracy, Average, Averaged, ClassScore, Kind
from strop.metrics.types import Metric

_AVERAGES: tuple[Average, ...] = ("macro", "micro", "weighted")
_KINDS: tuple[Kind, ...] = ("precision", "recall", "f1")


@dataclass(frozen=True)
class MetricContext:
    """What metric constructors need to know about the task."""

    labels: tuple[str, ...] = ()
    rules: Mapping[str, RuleSpec] = field(default_factory=dict)


Factory = Callable[[MetricContext], Metric]
ParamFactory = Callable[[str, MetricContext], Metric]


def _averaged_factory(average: Average, kind: Kind) -> Factory:
    return lambda ctx: Averaged(average, kind, ctx.labels)


def _simple_factories() -> dict[str, Factory]:
    factories: dict[str, Factory] = {
        "accuracy": lambda ctx: Accuracy(ctx.labels),
        "exact_match": lambda ctx: ExactMatch(),
        "valid_output_rate": lambda ctx: ValidOutputRate(),
    }
    for average in _AVERAGES:
        for kind in _KINDS:
            factories[f"{average}_{kind}"] = _averaged_factory(average, kind)
    return factories


def _class_factory(kind: Kind) -> ParamFactory:
    def build(label: str, ctx: MetricContext) -> Metric:
        if ctx.labels and label not in ctx.labels:
            raise ConfigError(
                f"metric '{kind}:{label}' names an unknown label; labels: {', '.join(ctx.labels)}"
            )
        return ClassScore(kind, label, ctx.labels)

    return build


_SIMPLE = _simple_factories()
_PARAMETERIZED: dict[str, ParamFactory] = {kind: _class_factory(kind) for kind in _KINDS}


def known_metric_names() -> list[str]:
    """Built-in metric names, plus the parameterized forms."""
    return sorted(_SIMPLE) + [f"{prefix}:<label>" for prefix in sorted(_PARAMETERIZED)]


def register_parameterized(prefix: str, factory: ParamFactory) -> None:
    """Add a metric family addressed as ``<prefix>:<argument>``."""
    _PARAMETERIZED[prefix] = factory


def register(name: str, factory: Factory) -> None:
    """Add a built-in metric addressed by ``name``."""
    _SIMPLE[name] = factory


def build_metric(name: str, context: MetricContext) -> Metric:
    """Create the metric called ``name`` or raise ``ConfigError`` listing the valid names."""
    if name in _SIMPLE:
        return _SIMPLE[name](context)
    prefix, separator, argument = name.partition(":")
    if separator and argument and prefix in _PARAMETERIZED:
        return _PARAMETERIZED[prefix](argument, context)
    raise ConfigError(f"unknown metric '{name}'; available: {', '.join(known_metric_names())}")
