from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SnapshotConfig(StrictModel):
    path: Path
    manifest_path: Path


class FeaturePolicyConfig(StrictModel):
    manifest_path: Path
    maximum_missing_rate: float = 0.95
    minimum_non_null_count: int = 50
    drop_zero_variance: bool = True
    drop_duplicate_vectors: bool = True

    @field_validator("maximum_missing_rate")
    @classmethod
    def validate_missing_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("maximum_missing_rate must be between 0 and 1")
        return value


class SelectionConfig(StrictModel):
    significant_delay_weight: float
    primary_metric: str = "weighted_macro_f1_significant_recall"

    @field_validator("significant_delay_weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("significant_delay_weight must be between 0 and 1")
        return value


class TuningConfig(StrictModel):
    strategy: str = "randomized_search"
    iterations: int = 20
    cross_validation_folds: int = 5
    n_estimators: List[int] = Field(default_factory=lambda: [250, 500, 800])
    max_depth: List[Optional[int]] = Field(default_factory=lambda: [6, 8, 12, 16, None])
    min_samples_leaf: List[int] = Field(default_factory=lambda: [5, 10, 15, 25])
    min_samples_split: List[int] = Field(default_factory=lambda: [10, 20, 40])
    max_features: List[Union[str, float]] = Field(
        default_factory=lambda: ["sqrt", 0.05, 0.10, 0.15]
    )
    max_samples: List[float] = Field(default_factory=lambda: [0.70, 0.85, 1.00])
    class_weight: List[Union[str, Dict[int, float]]] = Field(
        default_factory=lambda: ["balanced", "balanced_subsample"]
    )
    criterion: List[str] = Field(default_factory=lambda: ["gini", "entropy", "log_loss"])
    missing_indicators: List[bool] = Field(default_factory=lambda: [False, True])

    @field_validator("iterations", "cross_validation_folds")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("tuning counts must be positive")
        return value


class SplitConfig(StrictModel):
    locked_holdout_fraction: float = 0.20
    temporal_test_fraction: float = 0.20
    split_seed: int = 42
    minimum_class_support: int = 2
    minimum_customer_rows: int = 20
    minimum_customer_class_support: int = 2
    run_temporal_test: bool = True
    temporal_date_column: str = "PLANNEDSTARTDATE"
    run_customer_tests: bool = True

    @field_validator("locked_holdout_fraction", "temporal_test_fraction")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if value <= 0.0 or value >= 1.0:
            raise ValueError("split fractions must be strictly between 0 and 1")
        return value


class ResourceConfig(StrictModel):
    model_n_jobs: int = -1
    maximum_concurrent_candidates: int = 1
    bootstrap_iterations: int = 1000
    benchmark_mode: bool = False

    @field_validator("maximum_concurrent_candidates", "bootstrap_iterations")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("resource counts must be positive")
        return value


class ReportingConfig(StrictModel):
    generate_html: bool = True
    top_candidate_count: int = 20
    calibration_bins: int = 10


class SubgroupEvaluationConfig(StrictModel):
    enabled: bool = False
    schema_version: str = "schedule-subgroups-v1"
    required_for_global_production: List[str] = Field(default_factory=lambda: [
        "planned_duration",
        "planned_value",
        "contract_item_count",
        "predictor_missingness",
    ])
    development_minimum_rows_per_band: int = 50
    development_minimum_rows_per_class_per_band: int = 4
    holdout_minimum_rows_per_band: int = 20
    holdout_minimum_rows_per_class_per_band: int = 2
    duration_boundaries_days: List[float] = Field(default_factory=lambda: [60.0, 365.0])
    planned_value_boundaries: List[float] = Field(
        default_factory=lambda: [0.0, 1_000_000.0, 10_000_000.0]
    )
    contract_item_boundaries: List[int] = Field(default_factory=lambda: [20, 50])
    missingness_mode: str = "zero_vs_nonzero"

    @model_validator(mode="after")
    def validate_subgroups(self):
        known = {
            "planned_duration", "planned_value",
            "contract_item_count", "predictor_missingness",
        }
        unknown = set(self.required_for_global_production) - known
        if unknown:
            raise ValueError("Unknown required subgroup families: {}".format(sorted(unknown)))
        if len(set(self.required_for_global_production)) != len(
            self.required_for_global_production
        ):
            raise ValueError("Required subgroup families must be unique")
        if self.duration_boundaries_days != sorted(set(self.duration_boundaries_days)):
            raise ValueError("Duration boundaries must be unique and increasing")
        if len(self.duration_boundaries_days) != 2:
            raise ValueError("Exactly two duration boundaries are required")
        if self.planned_value_boundaries != sorted(set(self.planned_value_boundaries)):
            raise ValueError("Planned-value boundaries must be unique and increasing")
        if len(self.planned_value_boundaries) != 3:
            raise ValueError("Exactly three planned-value boundaries are required")
        if self.contract_item_boundaries != sorted(set(self.contract_item_boundaries)):
            raise ValueError("Contract-item boundaries must be unique and increasing")
        if len(self.contract_item_boundaries) != 2:
            raise ValueError("Exactly two contract-item boundaries are required")
        if self.missingness_mode != "zero_vs_nonzero":
            raise ValueError("Only zero_vs_nonzero missingness is supported")
        for population, total, per_class in (
            ("development", self.development_minimum_rows_per_band,
             self.development_minimum_rows_per_class_per_band),
            ("holdout", self.holdout_minimum_rows_per_band,
             self.holdout_minimum_rows_per_class_per_band),
        ):
            if total <= 0 or per_class <= 0:
                raise ValueError("{} subgroup minimums must be positive".format(population))
            if per_class * 3 > total:
                raise ValueError(
                    "{} total minimum cannot be smaller than three class minimums".format(
                        population
                    )
                )
        return self


class CustomerModelConfig(StrictModel):
    enabled: bool = False
    tuning_development_minimum_rows: int = 50
    tuning_development_minimum_rows_per_class: int = 4
    tuning_holdout_minimum_rows: int = 20
    tuning_holdout_minimum_rows_per_class: int = 2
    absolute_development_minimum_rows: int = 30
    absolute_development_minimum_rows_per_class: int = 2
    absolute_holdout_minimum_rows: int = 10
    absolute_holdout_minimum_rows_per_class: int = 2
    fallback_hyperparameters: str = "selected_all_customer"
    performance_policy: str = "same_as_all_customer"

    @model_validator(mode="after")
    def validate_customer_models(self):
        if self.fallback_hyperparameters != "selected_all_customer":
            raise ValueError("Only selected_all_customer fallback is supported")
        if self.performance_policy != "same_as_all_customer":
            raise ValueError("Only same_as_all_customer performance policy is supported")
        pairs = (
            ("tuning development", self.tuning_development_minimum_rows,
             self.tuning_development_minimum_rows_per_class),
            ("tuning holdout", self.tuning_holdout_minimum_rows,
             self.tuning_holdout_minimum_rows_per_class),
            ("absolute development", self.absolute_development_minimum_rows,
             self.absolute_development_minimum_rows_per_class),
            ("absolute holdout", self.absolute_holdout_minimum_rows,
             self.absolute_holdout_minimum_rows_per_class),
        )
        for name, total, per_class in pairs:
            if total <= 0 or per_class <= 0:
                raise ValueError("{} minimums must be positive".format(name))
            if per_class * 3 > total:
                raise ValueError(
                    "{} total minimum cannot be smaller than three class minimums".format(name)
                )
        if self.absolute_development_minimum_rows > self.tuning_development_minimum_rows:
            raise ValueError("Absolute development floor cannot exceed tuning floor")
        if self.absolute_holdout_minimum_rows > self.tuning_holdout_minimum_rows:
            raise ValueError("Absolute holdout floor cannot exceed tuning floor")
        if self.absolute_development_minimum_rows_per_class > self.tuning_development_minimum_rows_per_class:
            raise ValueError("Absolute development class floor cannot exceed tuning floor")
        if self.absolute_holdout_minimum_rows_per_class > self.tuning_holdout_minimum_rows_per_class:
            raise ValueError("Absolute holdout class floor cannot exceed tuning floor")
        return self


class TrainingRunConfig(StrictModel):
    schema_version: str = "schedule-training-run-v1"
    run_name: str
    random_seed: int
    training_snapshot: SnapshotConfig
    labels: SnapshotConfig
    target_definition_version: str
    feature_policy: FeaturePolicyConfig
    selection: SelectionConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    subgroup_evaluation: SubgroupEvaluationConfig = Field(
        default_factory=SubgroupEvaluationConfig
    )
    customer_models: CustomerModelConfig = Field(default_factory=CustomerModelConfig)
    release_policy_path: Path
    output_root: Path

    @field_validator("run_name")
    @classmethod
    def validate_run_name(cls, value: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if not value or any(character not in allowed for character in value):
            raise ValueError("run_name must be nonempty and filesystem-safe")
        return value


def _resolve_paths(config: TrainingRunConfig, base: Path) -> TrainingRunConfig:
    path_fields = [
        (config.training_snapshot, "path"),
        (config.training_snapshot, "manifest_path"),
        (config.labels, "path"),
        (config.labels, "manifest_path"),
        (config.feature_policy, "manifest_path"),
        (config, "release_policy_path"),
        (config, "output_root"),
    ]
    for owner, field_name in path_fields:
        value = getattr(owner, field_name)
        if not value.is_absolute():
            object.__setattr__(owner, field_name, (base / value).resolve())
    return config


def load_run_config(path: Path) -> TrainingRunConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        config = TrainingRunConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError("Invalid training run configuration: {}".format(exc)) from exc
    config = _resolve_paths(config, Path.cwd())
    required = [
        config.training_snapshot.path,
        config.training_snapshot.manifest_path,
        config.labels.path,
        config.labels.manifest_path,
        config.feature_policy.manifest_path,
        config.release_policy_path,
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise ConfigurationError("Required input paths do not exist: " + ", ".join(missing))
    return config
