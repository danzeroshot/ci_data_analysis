-- Batch inference extraction template.
-- The MCP server should create/populate REQUESTED_SCHEDULE_PROJECTS with bound
-- values or Snowflake connector bulk insertion; never interpolate raw IDs into SQL.

CREATE OR REPLACE TEMPORARY TABLE REQUESTED_SCHEDULE_PROJECTS (
    CustomerName STRING NOT NULL,
    ProjectID STRING NOT NULL
);

-- Example only. Use bound parameters/bulk insert in the MCP implementation.
-- INSERT INTO REQUESTED_SCHEDULE_PROJECTS (CustomerName, ProjectID)
-- SELECT column1, column2 FROM VALUES ('UDOT', '1425'), ('Lincoln', '12345');

SELECT
    requested.CustomerName AS RequestedCustomerName,
    requested.ProjectID AS RequestedProjectID,
    IFF(features.ProjectID IS NULL, 'FEATURE_ROW_NOT_FOUND', NULL) AS ExtractionErrorCode,
    features.CustomerName,
    features.ProjectID,
    features.FeatureAsOfUtc,
    features.FeatureSchemaVersion,
    features.KeywordManifestVersion,
    features.RefreshRunId,
    features.RefreshStartedAtUtc,
    /* <MODEL_FEATURE_COLUMN_LIST_FROM_TRUSTED_SCHEMA_ARTIFACT> */
    DATEDIFF(hour, features.FeatureAsOfUtc, CURRENT_TIMESTAMP()) AS FeatureAgeHours,
    IFF(
        features.FeatureAsOfUtc IS NOT NULL
        AND DATEDIFF(hour, features.FeatureAsOfUtc, CURRENT_TIMESTAMP()) <= COALESCE(
            (
                SELECT TRY_TO_NUMBER(CONFIG_VALUE)
                FROM <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_FEATURE_STORE_CONFIG
                WHERE CONFIG_KEY = 'MAX_FEATURE_AGE_HOURS'
            ),
            24
        ),
        TRUE,
        FALSE
    ) AS IsFeatureFresh
FROM REQUESTED_SCHEDULE_PROJECTS requested
LEFT JOIN <ANALYTICS_DATABASE>.ML_FEATURES.SCHEDULE_PROJECT_FEATURES_CURRENT features
    ON features.CustomerName = requested.CustomerName
   AND TO_VARCHAR(features.ProjectID) = requested.ProjectID
ORDER BY requested.CustomerName, requested.ProjectID;

-- The MCP repository replaces the MODEL_FEATURE_COLUMN_LIST placeholder with
-- identifiers from its packaged schedule_risk_feature_schema.json. It must not
-- accept column names from request input. This avoids transferring raw text and
-- store-only audit fields and preserves the exact trained feature order.
-- Remember to include a comma after the final projected model feature.
--
-- The MCP layer must reject stale rows with FEATURE_DATA_STALE when
-- IsFeatureFresh = FALSE. Stale data is not recalculated synchronously.

