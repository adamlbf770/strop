"""Console construction with an explicit color mode."""

from enum import StrEnum

from rich.console import Console


class ColorMode(StrEnum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


def make_console(mode: ColorMode = ColorMode.AUTO) -> Console:
    """A stdout console. ``auto`` colors only on a terminal and respects ``NO_COLOR``."""
    if mode is ColorMode.ALWAYS:
        return Console(force_terminal=True, color_system="standard", highlight=False)
    if mode is ColorMode.NEVER:
        return Console(force_terminal=False, no_color=True, color_system=None, highlight=False)
    return Console(highlight=False)
