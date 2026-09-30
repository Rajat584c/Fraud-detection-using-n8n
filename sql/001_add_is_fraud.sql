-- Adds the ground-truth label used for retraining.
-- NULL = not yet investigated, TRUE = confirmed fraud, FALSE = confirmed genuine.
-- Nothing in the n8n workflow writes this column, so existing inserts keep working.
ALTER TABLE claims ADD COLUMN IF NOT EXISTS is_fraud BOOLEAN;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS labelled_at TIMESTAMPTZ;

-- Example: an investigator closes a case
-- UPDATE claims SET is_fraud = TRUE,  labelled_at = NOW() WHERE claim_id = 'CLM-1042';
-- UPDATE claims SET is_fraud = FALSE, labelled_at = NOW() WHERE claim_id = 'CLM-1043';
