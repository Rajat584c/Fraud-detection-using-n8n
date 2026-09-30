"""Settings, all overridable through environment variables."""
import os


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "y")


# Paths
DATA_PATH = os.getenv("DATA_PATH", "/app/data/insurance_claims.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

# Postgres (same database n8n writes to). Leave empty to retrain on Kaggle data only.
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Kaggle total_claim_amount is multiplied by this to match the currency/scale of your form.
AMOUNT_SCALE = _float("AMOUNT_SCALE", 1.0)

# Keyword prior: the Kaggle data has no description text, so keyword features are
# seeded with a small number of copied rows (see features.py). Turn off once you
# have enough confirmed claims with descriptions.
KEYWORD_PRIOR = _bool("KEYWORD_PRIOR", True)

# Confirmed claims from Postgres count this many times more than a Kaggle row.
LABEL_WEIGHT = _float("LABEL_WEIGHT", 3.0)

# Risk bands on the 0-100 fraud_score (same cut-offs as the old JavaScript rules).
MEDIUM_THRESHOLD = _float("MEDIUM_THRESHOLD", 30)
HIGH_THRESHOLD = _float("HIGH_THRESHOLD", 60)

# A retrained model is rejected if its test ROC-AUC is this much worse than the
# live model, unless the request passes force=true.
MAX_AUC_DROP = _float("MAX_AUC_DROP", 0.02)

# Optional shared secret for /retrain (sent as the X-API-Key header).
RETRAIN_API_KEY = os.getenv("RETRAIN_API_KEY", "")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
