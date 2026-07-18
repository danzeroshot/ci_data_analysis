-- Training extraction boundary for the Schedule Risk Agent.
-- This query intentionally keeps retrospective labels separate from the
-- beginning-feature store. Replace <TRAINING_SNAPSHOT_TABLE> with the immutable
-- snapshot retained for the model version being trained.

WITH training_features AS (
    SELECT *
    FROM <TRAINING_SNAPSHOT_TABLE>
),
delay_targets AS (
    -- Populate using the payment_rows/project_delay_targets logic from
    -- custpaydetails_project_feature_table_with_keywords_materialized.sql.
    -- The target branch must never be joined into CURRENT inference features.
    SELECT
        CustomerName,
        ProjectID,
        PercentDelayed,
        CASE
            WHEN PercentDelayed <= 0 THEN 0
            WHEN PercentDelayed <= 25 THEN 1
            ELSE 2
        END AS ScheduleRiskBin
    FROM <SCHEDULE_DELAY_TARGET_TABLE>
    WHERE PercentDelayed IS NOT NULL
)
SELECT
    features.*,
    targets.PercentDelayed,
    targets.ScheduleRiskBin
FROM training_features features
INNER JOIN delay_targets targets
    ON targets.CustomerName = features.CustomerName
   AND targets.ProjectID = features.ProjectID;

