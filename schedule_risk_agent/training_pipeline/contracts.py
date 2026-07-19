from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


CLASS_IDS = (0, 1, 2)
CLASS_LABELS = ("no_delay", "mild_delay", "significant_delay")
TARGET_COLUMN = "PERCENTDELAYED"
LABEL_COLUMN = "SCHEDULERISKBIN"
KEY_COLUMNS = ("CUSTOMERNAME", "PROJECTID")


class TrainingPipelineError(Exception):
    code = "training_pipeline_error"


class ConfigurationError(TrainingPipelineError):
    code = "configuration_error"


class DataContractError(TrainingPipelineError):
    code = "data_contract_error"


class StageError(TrainingPipelineError):
    code = "stage_error"


class ReleaseGateError(TrainingPipelineError):
    code = "release_gate_error"


@dataclass(frozen=True)
class MetricValue:
    value: Optional[float]
    undefined_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "undefined_reason": self.undefined_reason,
        }
