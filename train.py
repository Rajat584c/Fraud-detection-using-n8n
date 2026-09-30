"""Train the first model from the command line (optional; the API also trains on startup).

    python train.py            # Kaggle CSV + labelled Postgres claims
    python train.py --no-db    # Kaggle CSV only
"""
import argparse
import json

from app import training

parser = argparse.ArgumentParser()
parser.add_argument("--no-db", action="store_true", help="skip Postgres labelled claims")
args = parser.parse_args()

model, meta = training.train(use_db=not args.no_db)
path = training.save(model, meta)
print(json.dumps({"saved_to": path, "test_metrics": meta["test_metrics"], "rows": meta["rows"],
                  "warnings": meta["warnings"]}, indent=2))
