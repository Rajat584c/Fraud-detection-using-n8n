# Claims Fraud ML Service (XGBoost)

This replaces the "Code in JavaScript" node in the `Fraud detection` n8n workflow. The new "ML Fraud Scoring" node is an HTTP Request node. It sends each claim to this service and gets back the same fields the Postgres insert node already uses. No other node in the workflow has changed.

```
Webhook -> Edit Fields -> ML Fraud Scoring (POST http://ml-api:8000/predict) -> Insert rows in a table
```

## What's in the folder

| Path | Purpose |
|---|---|
| `n8n/Fraud detection.json` | Your workflow with only the scoring node swapped |
| `n8n/Fraud model retrain (weekly).json` | Optional separate workflow that calls `/retrain` every Sunday at 2 AM |
| `app/` | FastAPI service: features, training, API |
| `train.py` | Train from the command line (optional, the API also trains on first start) |
| `sql/001_add_is_fraud.sql` | Adds the `is_fraud` label column used for retraining |
| `Dockerfile`, `docker-compose.yml`, `.env.example` | Run it next to n8n |

## Setup

**1. Get the training data.** Download `insurance_claims.csv` from
https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data and put it in `data/`.

**2. Add the label column** (run once in your Postgres):

```bash
psql "<your connection string>" -f sql/001_add_is_fraud.sql
```

**3. Configure.** `cp .env.example .env` and set `DATABASE_URL` to the database n8n writes to.

**4. Put n8n and the service on one Docker network:**

```bash
docker network create fraud-net
docker network connect fraud-net <your-n8n-container-name>    # find it with: docker ps
docker network connect fraud-net <your-postgres-container>    # only if Postgres runs in Docker
```

**5. Start the service:**

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

The first start trains a model from the Kaggle file and saves it in the `ml_models` volume. After a restart the service loads the saved model and skips training.

**6. Import the workflow** in n8n (Import from File, pick `n8n/Fraud detection.json`). Re-select your Postgres credential if n8n asks, then activate it.

> If you would rather not use a shared network, change the node URL to `http://host.docker.internal:8000/predict`. On Linux you also need `--add-host=host.docker.internal:host-gateway` on the n8n container.

## API

### `POST /predict`
Input is the `body` object from Edit Fields:

```json
{"claimId":"CLM-3","policyId":"POL-9","claimAmount":"110000","claimType":"major",
 "claimDescription":"accident looked staged, fake bills"}
```

Output (the Postgres node reads the first nine keys; it ignores the last two):

```json
{"claimId":"CLM-3","policyId":"POL-9","claimAmount":110000.0,"claimType":"major",
 "claimDescription":"accident looked staged, fake bills",
 "fraud_score":97,"risk_level":"HIGH","status":"UNDER REVIEW",
 "fraud_flags":"Suspicious description: staged, fake, High claim amount (110,000), Major claim",
 "fraud_probability":0.9712,"model_version":"20260923T101116365Z"}
```

- `fraud_score` is the model's fraud probability times 100, rounded.
- `risk_level` is LOW below 30, MEDIUM from 30 to 59 and HIGH from 60 up. These are the cut-offs your old rules used, and you can change them in `.env`.
- `status` is `UNDER REVIEW` when the risk is HIGH, otherwise `PROCESSING`.
- `fraud_flags` lists the features that pushed this particular claim toward fraud, largest first. They come from XGBoost's SHAP contributions, so they match what the model actually did.

### `POST /retrain`
Retrains on Kaggle plus every claim where `is_fraud` is set, then swaps the new model in without a restart.

```bash
curl -X POST http://localhost:8000/retrain -H "X-API-Key: <RETRAIN_API_KEY>"
curl -X POST http://localhost:8000/retrain -H "Content-Type: application/json" -d '{"force": true}'
```

If the new model's test ROC-AUC is more than `MAX_AUC_DROP` (0.02) below the live model's, it is not promoted and the response explains why. `force: true` promotes it anyway. Every version is kept in `/models/<version>/` with a `meta.json` holding its metrics, so you can roll back by editing `/models/CURRENT`.

### `GET /health`
Shows the live model version, test metrics, row counts and any training warnings.

## How retraining gets smarter

The model can only learn from claims whose real outcome is known. When an investigator closes a case:

```sql
UPDATE claims SET is_fraud = TRUE,  labelled_at = NOW() WHERE claim_id = 'CLM-1042';
UPDATE claims SET is_fraud = FALSE, labelled_at = NOW() WHERE claim_id = 'CLM-1043';
```

Each confirmed claim counts 3 times as much as a Kaggle row (`LABEL_WEIGHT`). The weekly retrain workflow picks them up automatically. Set the same `X-API-Key` in that workflow as in `.env`, or delete the header if you leave the key empty.

## Model details

**Features** (all built from the three form fields):

| Feature | From the form | From Kaggle |
|---|---|---|
| `claim_amount` | claimAmount | total_claim_amount x `AMOUNT_SCALE` |
| `is_major` | claimType = "major" | incident_severity is Major Damage or Total Loss |
| `kw_explicit_fraud` | staged, fake, fraud, fabricated, false | not available (see keyword prior) |
| `kw_duplicate` | duplicate, already claimed | not available (see keyword prior) |
| `kw_no_evidence` | "no evidence" | no police report and 0 witnesses (proxy) |

**Algorithm.** XGBoost, 200 trees of depth 2, `scale_pos_weight` for the roughly 25% fraud rate. Monotone constraints mean every feature can only raise the score, so a larger amount or a suspicious keyword never lowers risk. That matters on a dataset of about 1,000 rows. 20% of the real rows are held out for the test metrics, and the model that goes live is then refit on all of them.

**Keyword prior.** The Kaggle data has no description text, so it cannot teach the keyword features anything. To keep descriptions counting from day one, training copies 20% of fraud rows and 2% of genuine rows with each keyword flag switched on. This encodes one assumption: that wording like this is about 10 times more common in fraud. The copies are never used for evaluation. Once you have a few hundred labelled claims, set `KEYWORD_PRIOR=false` so the model relies only on real evidence.

## Things to know

- **Currency.** Kaggle amounts are in USD and range from about 100 to 115,000. If your form takes INR, set `AMOUNT_SCALE` (for example `83`) so amounts are on the same scale. Otherwise the model will read almost every INR claim as "very large".
- **Expected accuracy.** With only amount and severity, the model is a modest first screen. On the Kaggle data, expect a test ROC-AUC of roughly 0.7 to 0.75 (check `/health` after the first start). It improves as labelled claims build up.
- **Probabilities are skewed upward.** Because of `scale_pos_weight`, the model leans toward flagging fraud, which is what you want for triage. Read `fraud_score` as a risk score, not a calibrated probability.
- **Existing workflow bug, left untouched.** In "Respond to Webhook", values like `"={{ $json.claim_id }}"` have an extra `=` inside the quotes, so the lookup API returns `"=CLM123"`. Removing the inner `=` fixes it.
