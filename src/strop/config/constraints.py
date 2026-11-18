"""Parsing of ``--constraint`` flags such as ``valid_output_rate>=0.98``."""

import re
from collections.abc import Iterable

from strop.config.metrics import Bound
from strop.errors import ConfigError

_EXPRESSION = re.compile(r"^\s*(?P<metric>[\w:.\-]+)\s*(?P<op>>=|<=)\s*(?P<value>[-+0-9.eE]+)\s*$")


def parse_constraints(expressions: Iterable[str]) -> dict[str, Bound]:
    """Turn ``metric>=x`` and ``metric<=y`` expressions into per-metric bounds.

    Two expressions for the same metric (one ``>=``, one ``<=``) are merged.
    """
    limits: dict[str, dict[str, float]] = {}
    for text in expressions:
        match = _EXPRESSION.match(text)
        if match is None:
            raise ConfigError(
                f"invalid constraint '{text}'; use the form 'metric>=0.9' or 'metric<=0.1'"
            )
        try:
            value = float(match["value"])
        except ValueError:
            raise ConfigError(f"invalid number in constraint '{text}'") from None
        side = "min" if match["op"] == ">=" else "max"
        bounds = limits.setdefault(match["metric"], {})
        if side in bounds:
            raise ConfigError(f"constraint on '{match['metric']}' sets '{side}' more than once")
        bounds[side] = value
    return {metric: Bound(**bounds) for metric, bounds in limits.items()}
