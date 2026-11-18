"""Where a project keeps its configuration and local state."""

from dataclasses import dataclass
from pathlib import Path

CONFIG_FILENAME = "strop.yaml"
STATE_DIRNAME = ".strop"


@dataclass(frozen=True)
class ProjectPaths:
    """Locations derived from the project root directory."""

    root: Path

    @property
    def config_file(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIRNAME

    @property
    def database(self) -> Path:
        return self.state_dir / "strop.db"

    def ensure(self) -> None:
        """Create the state directory if it does not exist."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
