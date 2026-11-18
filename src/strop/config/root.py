"""The complete project configuration stored in strop.yaml."""

from pathlib import Path

from pydantic import Field

from strop.config.base import StrictModel
from strop.config.dataset import DatasetConfig
from strop.config.limits import BudgetsConfig, ExecutionConfig
from strop.config.llm import ModelsConfig
from strop.config.metrics import MetricsConfig
from strop.config.optimization import OptimizationConfig
from strop.config.output import OutputConfig
from strop.config.split import SplitConfig


class ProjectConfig(StrictModel):
    name: str = Field(min_length=1)
    task_description: str = Field(min_length=1)


class PromptConfig(StrictModel):
    file: Path = Path("prompt.yaml")


class StropConfig(StrictModel):
    project: ProjectConfig
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    dataset: DatasetConfig
    output: OutputConfig
    models: ModelsConfig
    metrics: MetricsConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
