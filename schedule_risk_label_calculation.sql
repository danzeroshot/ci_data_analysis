-- Schedule Risk Agent retrospective label calculation.
-- This query uses temporary tables only and does not require persistent DDL privileges.
-- It intentionally remains separate from the beginning-only feature calculation.
-- Target definition version: schedule-delay-v1.

CREATE OR REPLACE TEMPORARY TABLE payment_rows AS
-- Amtrak
    SELECT
        'Amtrak' AS CustomerName,
        PM.ProjectID AS ProjectID,
        CM.StartDt AS ContractStartDate,
        CM.CLOSUREDT AS ContractClosureDate,
        WP.POSTINGDATE AS WPPostingDate,
        CI.UnitPrice * PED.Quantity AS CalculatedWorkCompletedAmount
    FROM AMTRAKAIML.DBO.CORITEMItemDetails CI
    INNER JOIN AMTRAKAIML.DBO.CONTMGTMaster CM
        ON CM.ID = CI.PARENTINSTANCEID
       AND CI.MODULEID = 'CONTMGT'
    INNER JOIN AMTRAKAIML.DBO.PROJECTPROJECTMAIN PM
        ON PM.ProjectID = CM.ProjectId
    INNER JOIN AMTRAKAIML.DBO.PROCMGTCommitments C
        ON C.POType = 'CONTMGT'
       AND C.POTypeInstanceID = CM.ID
    INNER JOIN AMTRAKAIML.DBO.PROCMGTWorkPosting WP
        ON WP.PayItemID = CI.ItemID
       AND WP.POID = C.POID
       AND WP.ReferencePostingType = 'ITMPOST'
    INNER JOIN AMTRAKAIML.DBO.PROCMGTPEDetails PED
        ON PED.WorkPostingID = WP.WPostingID
       AND PED.WorkItemID = CI.ItemID
    INNER JOIN AMTRAKAIML.DBO.PROCMGTPayEstimates PE
        ON PE.PEID = PED.PEID
    WHERE PM.PROJECTNAME IS NOT NULL
      AND WP.STATUS IS NOT NULL
      AND PE.STATUS IS NOT NULL
      AND WP.POSTINGDATE IS NOT NULL

    UNION ALL

    -- UDOT
    SELECT
        'UDOT' AS CustomerName,
        PM.ProjectID AS ProjectID,
        CM.StartDt AS ContractStartDate,
        CM.CLOSUREDT AS ContractClosureDate,
        WP.POSTINGDATE AS WPPostingDate,
        CI.UnitPrice * PED.Quantity AS CalculatedWorkCompletedAmount
    FROM UDOTAIML.DBO.CORITEMItemDetails CI
    INNER JOIN UDOTAIML.DBO.CONTMGTMaster CM
        ON CM.ID = CI.PARENTINSTANCEID
       AND CI.MODULEID = 'CONTMGT'
    INNER JOIN UDOTAIML.DBO.PROJECTPROJECTMAIN PM
        ON PM.ProjectID = CM.ProjectId
    INNER JOIN UDOTAIML.DBO.PROCMGTCommitments C
        ON C.POType = 'CONTMGT'
       AND C.POTypeInstanceID = CM.ID
    INNER JOIN UDOTAIML.DBO.PROCMGTWorkPosting WP
        ON WP.PayItemID = CI.ItemID
       AND WP.POID = C.POID
       AND WP.ReferencePostingType = 'ITMPOST'
    INNER JOIN UDOTAIML.DBO.PROCMGTPEDetails PED
        ON PED.WorkPostingID = WP.WPostingID
       AND PED.WorkItemID = CI.ItemID
    INNER JOIN UDOTAIML.DBO.PROCMGTPayEstimates PE
        ON PE.PEID = PED.PEID
    WHERE PM.PROJECTNAME IS NOT NULL
      AND WP.STATUS IS NOT NULL
      AND PE.STATUS IS NOT NULL
      AND WP.POSTINGDATE IS NOT NULL

    UNION ALL

    -- CLV
    SELECT
        'CLV' AS CustomerName,
        PM.ProjectID AS ProjectID,
        CM.StartDt AS ContractStartDate,
        CM.CLOSUREDT AS ContractClosureDate,
        WP.POSTINGDATE AS WPPostingDate,
        CI.UnitPrice * PED.Quantity AS CalculatedWorkCompletedAmount
    FROM CLVAIML.DBO.CORITEMItemDetails CI
    INNER JOIN CLVAIML.DBO.CONTMGTMaster CM
        ON CM.ID = CI.PARENTINSTANCEID
       AND CI.MODULEID = 'CONTMGT'
    INNER JOIN CLVAIML.DBO.PROJECTPROJECTMAIN PM
        ON PM.ProjectID = CM.ProjectId
    INNER JOIN CLVAIML.DBO.PROCMGTCommitments C
        ON C.POType = 'CONTMGT'
       AND C.POTypeInstanceID = CM.ID
    INNER JOIN CLVAIML.DBO.PROCMGTWorkPosting WP
        ON WP.PayItemID = CI.ItemID
       AND WP.POID = C.POID
       AND WP.ReferencePostingType = 'ITMPOST'
    INNER JOIN CLVAIML.DBO.PROCMGTPEDetails PED
        ON PED.WorkPostingID = WP.WPostingID
       AND PED.WorkItemID = CI.ItemID
    INNER JOIN CLVAIML.DBO.PROCMGTPayEstimates PE
        ON PE.PEID = PED.PEID
    WHERE PM.PROJECTNAME IS NOT NULL
      AND WP.STATUS IS NOT NULL
      AND PE.STATUS IS NOT NULL
      AND WP.POSTINGDATE IS NOT NULL

    UNION ALL

    -- CCD
    SELECT
        'CCD' AS CustomerName,
        PM.ProjectID AS ProjectID,
        CM.StartDt AS ContractStartDate,
        CM.CLOSUREDT AS ContractClosureDate,
        WP.POSTINGDATE AS WPPostingDate,
        CI.UnitPrice * PED.Quantity AS CalculatedWorkCompletedAmount
    FROM CCDAIML.DBO.CORITEMItemDetails CI
    INNER JOIN CCDAIML.DBO.CONTMGTMaster CM
        ON CM.ID = CI.PARENTINSTANCEID
       AND CI.MODULEID = 'CONTMGT'
    INNER JOIN CCDAIML.DBO.PROJECTPROJECTMAIN PM
        ON PM.ProjectID = CM.ProjectId
    INNER JOIN CCDAIML.DBO.PROCMGTCommitments C
        ON C.POType = 'CONTMGT'
       AND C.POTypeInstanceID = CM.ID
    INNER JOIN CCDAIML.DBO.PROCMGTWorkPosting WP
        ON WP.PayItemID = CI.ItemID
       AND WP.POID = C.POID
       AND WP.ReferencePostingType = 'ITMPOST'
    INNER JOIN CCDAIML.DBO.PROCMGTPEDetails PED
        ON PED.WorkPostingID = WP.WPostingID
       AND PED.WorkItemID = CI.ItemID
    INNER JOIN CCDAIML.DBO.PROCMGTPayEstimates PE
        ON PE.PEID = PED.PEID
    WHERE PM.PROJECTNAME IS NOT NULL
      AND WP.STATUS IS NOT NULL
      AND PE.STATUS IS NOT NULL
      AND WP.POSTINGDATE IS NOT NULL

    UNION ALL

    -- Adams
    SELECT
        'Adams' AS CustomerName,
        PM.ProjectID AS ProjectID,
        CM.StartDt AS ContractStartDate,
        CM.CLOSUREDT AS ContractClosureDate,
        WP.POSTINGDATE AS WPPostingDate,
        CI.UnitPrice * PED.Quantity AS CalculatedWorkCompletedAmount
    FROM ADAMSAIML.DBO.CORITEMItemDetails CI
    INNER JOIN ADAMSAIML.DBO.CONTMGTMaster CM
        ON CM.ID = CI.PARENTINSTANCEID
       AND CI.MODULEID = 'CONTMGT'
    INNER JOIN ADAMSAIML.DBO.PROJECTPROJECTMAIN PM
        ON PM.ProjectID = CM.ProjectId
    INNER JOIN ADAMSAIML.DBO.PROCMGTCommitments C
        ON C.POType = 'CONTMGT'
       AND C.POTypeInstanceID = CM.ID
    INNER JOIN ADAMSAIML.DBO.PROCMGTWorkPosting WP
        ON WP.PayItemID = CI.ItemID
       AND WP.POID = C.POID
       AND WP.ReferencePostingType = 'ITMPOST'
    INNER JOIN ADAMSAIML.DBO.PROCMGTPEDetails PED
        ON PED.WorkPostingID = WP.WPostingID
       AND PED.WorkItemID = CI.ItemID
    INNER JOIN ADAMSAIML.DBO.PROCMGTPayEstimates PE
        ON PE.PEID = PED.PEID
    WHERE PM.PROJECTNAME IS NOT NULL
      AND WP.STATUS IS NOT NULL
      AND PE.STATUS IS NOT NULL
      AND WP.POSTINGDATE IS NOT NULL

    UNION ALL

    -- Lincoln
    SELECT
        'Lincoln' AS CustomerName,
        PM.ProjectID AS ProjectID,
        CM.StartDt AS ContractStartDate,
        CM.CLOSUREDT AS ContractClosureDate,
        WP.POSTINGDATE AS WPPostingDate,
        CI.UnitPrice * PED.Quantity AS CalculatedWorkCompletedAmount
    FROM CITYOFLINCOLNAIML.DBO.CORITEMItemDetails CI
    INNER JOIN CITYOFLINCOLNAIML.DBO.CONTMGTMaster CM
        ON CM.ID = CI.PARENTINSTANCEID
       AND CI.MODULEID = 'CONTMGT'
    INNER JOIN CITYOFLINCOLNAIML.DBO.PROJECTPROJECTMAIN PM
        ON PM.ProjectID = CM.ProjectId
    INNER JOIN CITYOFLINCOLNAIML.DBO.PROCMGTCommitments C
        ON C.POType = 'CONTMGT'
       AND C.POTypeInstanceID = CM.ID
    INNER JOIN CITYOFLINCOLNAIML.DBO.PROCMGTWorkPosting WP
        ON WP.PayItemID = CI.ItemID
       AND WP.POID = C.POID
       AND WP.ReferencePostingType = 'ITMPOST'
    INNER JOIN CITYOFLINCOLNAIML.DBO.PROCMGTPEDetails PED
        ON PED.WorkPostingID = WP.WPostingID
       AND PED.WorkItemID = CI.ItemID
    INNER JOIN CITYOFLINCOLNAIML.DBO.PROCMGTPayEstimates PE
        ON PE.PEID = PED.PEID
    WHERE PM.PROJECTNAME IS NOT NULL
      AND WP.STATUS IS NOT NULL
      AND PE.STATUS IS NOT NULL
      AND WP.POSTINGDATE IS NOT NULL
;

-- Optional sanity check while tuning: SELECT COUNT(*) AS payment_rows_rows FROM payment_rows;

CREATE OR REPLACE TEMPORARY TABLE project_delay_targets AS
SELECT
        CustomerName,
        ProjectID,
        MIN(ContractStartDate) AS TargetPlannedStartDate,
        MAX(ContractClosureDate) AS TargetPlannedEndDate,
        DATEADD(day, -30, MIN(WPPostingDate)) AS TargetActualStartDate,
        MAX(WPPostingDate) AS TargetActualEndDate,
        MIN(WPPostingDate) AS TargetFirstPostingDate,
        MAX(WPPostingDate) AS TargetLastPostingDate,
        DATEDIFF(day, MIN(ContractStartDate), MAX(ContractClosureDate)) AS TargetPlannedDurationDays,
        DATEDIFF(day, DATEADD(day, -30, MIN(WPPostingDate)), MAX(WPPostingDate)) AS TargetActualDurationDays,
        DATEDIFF(day, MIN(WPPostingDate), MAX(WPPostingDate)) AS TargetPaymentSpanDays,
        IFF(
            MIN(ContractStartDate) IS NOT NULL
            AND MAX(ContractClosureDate) IS NOT NULL
            AND DATEDIFF(day, MIN(ContractStartDate), MAX(ContractClosureDate)) > 0
            AND MIN(WPPostingDate) IS NOT NULL
            AND MAX(WPPostingDate) IS NOT NULL,
            100.0 * DATEDIFF(day, DATEADD(day, -30, MIN(WPPostingDate)), MAX(WPPostingDate))
                / NULLIF(DATEDIFF(day, MIN(ContractStartDate), MAX(ContractClosureDate)), 0) - 100.0,
            NULL
        ) AS PercentDelayed,
        SUM(CalculatedWorkCompletedAmount) AS TargetValidPostedProjectWorkCompletedAmount,
        COUNT(*) AS TargetValidPostedDetailRows,
        COUNT(DISTINCT WPPostingDate) AS TargetValidPostedPayDateCount
    FROM payment_rows
    GROUP BY CustomerName, ProjectID
;

SELECT
    CustomerName,
    ProjectID,
    TargetPlannedStartDate,
    TargetPlannedEndDate,
    TargetActualStartDate,
    TargetActualEndDate,
    TargetFirstPostingDate,
    TargetLastPostingDate,
    TargetPlannedDurationDays,
    TargetActualDurationDays,
    TargetPaymentSpanDays,
    PercentDelayed,
    CASE
        WHEN TargetPlannedStartDate IS NULL THEN 'missing_planned_start'
        WHEN TargetPlannedEndDate IS NULL THEN 'missing_planned_end'
        WHEN TargetPlannedDurationDays <= 0 THEN 'nonpositive_planned_duration'
        WHEN TargetActualStartDate IS NULL OR TargetActualEndDate IS NULL THEN 'missing_valid_posting_date'
        WHEN TargetActualDurationDays < 0 THEN 'actual_end_before_actual_start'
        WHEN PercentDelayed IS NULL THEN 'invalid_percent_delayed'
        ELSE NULL
    END AS LabelExclusionReason,
    CASE
        WHEN PercentDelayed <= 0 THEN 0
        WHEN PercentDelayed <= 25 THEN 1
        WHEN PercentDelayed IS NOT NULL THEN 2
        ELSE NULL
    END AS ScheduleRiskBin,
    'schedule-delay-v1' AS TargetDefinitionVersion,
    CURRENT_TIMESTAMP() AS TargetCalculatedAtUtc
FROM project_delay_targets
ORDER BY CustomerName, ProjectID;
