class ScheduleRiskError(Exception):
    code = "INTERNAL_ERROR"
    retryable = False

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


class ConfigurationError(ScheduleRiskError):
    code = "CONFIGURATION_ERROR"


class FeatureValidationError(ScheduleRiskError):
    code = "FEATURE_VALIDATION_FAILED"


class FeatureRepositoryNotReady(ScheduleRiskError):
    code = "FEATURE_REPOSITORY_NOT_READY"
    retryable = True


class FeatureDataStale(ScheduleRiskError):
    code = "FEATURE_DATA_STALE"
    retryable = True


class FeatureVersionMismatch(ScheduleRiskError):
    code = "FEATURE_VERSION_MISMATCH"


class FeatureRowNotFound(ScheduleRiskError):
    code = "FEATURE_ROW_NOT_FOUND"

