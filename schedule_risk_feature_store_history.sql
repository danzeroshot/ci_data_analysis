-- Optional operational history path. This file is intentionally independent
-- from current-table publication and can be omitted without changing inference.
-- Default retention is configured as three months.

CREATE TABLE IF NOT EXISTS
    <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_PROJECT_FEATURES_HISTORY
LIKE <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_PROJECT_FEATURES_CURRENT;

-- Make retries idempotent at refresh-run grain.
INSERT INTO <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_PROJECT_FEATURES_HISTORY
SELECT current_features.*
FROM <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_PROJECT_FEATURES_CURRENT current_features
WHERE NOT EXISTS (
    SELECT 1
    FROM <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_PROJECT_FEATURES_HISTORY history
    WHERE history.RefreshRunId = current_features.RefreshRunId
);

DELETE FROM <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_PROJECT_FEATURES_HISTORY
WHERE FeatureAsOfUtc < DATEADD(
    month,
    -COALESCE(
        (
            SELECT TRY_TO_NUMBER(CONFIG_VALUE)
            FROM <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_FEATURE_STORE_CONFIG
            WHERE CONFIG_KEY = 'HISTORY_RETENTION_MONTHS'
        ),
        3
    ),
    CURRENT_TIMESTAMP()
);

-- This rolling history is for audit and operational troubleshooting. A model's
-- immutable training snapshot must be retained with the model artifact and must
-- not depend on this three-month retention window.

