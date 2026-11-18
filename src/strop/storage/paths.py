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

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.state_dir / "reports"

    def ensure(self) -> None:
        """Create the state directories if they do not exist."""
        for directory in (self.state_dir, self.runs_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
