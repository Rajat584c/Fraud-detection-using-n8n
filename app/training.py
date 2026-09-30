"""Train, evaluate and save an XGBoost fraud model."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from . import config
from .features import FEATURES, MONOTONE, add_keyword_prior, frame_from_claims, frame_from_kaggle

log = logging.getLogger("training")


def fetch_labelled_claims() -> tuple[list[dict], str | None]:
    """Confirmed claims from Postgres (is_fraud not null). Returns (rows, warning)."""
    if not config.DATABASE_URL:
        return [], "DATABASE_URL not set, trained on Kaggle data only"
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(config.DATABASE_URL, connect_timeout=5, row_factory=dict_row) as conn:
            rows = conn.execute(
                "SELECT claim_amount, claim_type, claim_description, is_fraud "
                "FROM claims WHERE is_fraud IS NOT NULL"
            ).fetchall()
        return rows, None
    except Exception as exc:  # DB down or column missing: still retrain on Kaggle
        log.warning("Could not read labelled claims: %s", exc)
        return [], f"Could not read labelled claims from Postgres ({exc.__class__.__name__}: {exc})"


def build_dataset(use_db: bool = True) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    frames = []
    if os.path.exists(config.DATA_PATH):
        frames.append(frame_from_kaggle(config.DATA_PATH))
    else:
        warnings.append(f"Kaggle file not found at {config.DATA_PATH}")

    if use_db:
        rows, warn = fetch_labelled_claims()
        if warn:
            warnings.append(warn)
        if rows:
            frames.append(frame_from_claims(rows))

    if not frames:
        raise RuntimeError("No training data: add data/insurance_claims.csv or label claims in Postgres.")
    return pd.concat(frames, ignore_index=True), warnings


def _metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    out = {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }
    if len(set(y)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        out["pr_auc"] = round(float(average_precision_score(y, p)), 4)
    return out


def _new_model(y: pd.Series) -> XGBClassifier:
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    return XGBClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=1.0,
        min_child_weight=10,  # small dataset: avoid chasing noise
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,  # balance fraud vs legitimate
        monotone_constraints=MONOTONE,
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )


def train(use_db: bool = True) -> tuple[XGBClassifier, dict]:
    """Returns (fitted model, metadata). Nothing is saved here."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    data, warnings = build_dataset(use_db)
    real = data[data["source"] != "prior"]
    if real["label"].nunique() < 2:
        raise RuntimeError("Training data needs both fraud and non-fraud examples.")

    # Hold out 20% of real rows for evaluation; prior copies only go to training,
    # and only if their source row is in the training part.
    train_idx, test_idx = train_test_split(
        real.index, test_size=0.2, stratify=real["label"], random_state=config.RANDOM_STATE
    )
    train_df = data.loc[train_idx]
    if config.KEYWORD_PRIOR:
        train_df = add_keyword_prior(train_df, rng)
    test_df = data.loc[test_idx]

    model = _new_model(train_df["label"])
    model.fit(train_df[FEATURES], train_df["label"], sample_weight=train_df["weight"])
    test_metrics = _metrics(test_df["label"].to_numpy(), model.predict_proba(test_df[FEATURES])[:, 1])

    # Refit on all real rows (+ prior) for the model that goes live.
    full = add_keyword_prior(data, rng) if config.KEYWORD_PRIOR else data
    final = _new_model(full["label"])
    final.fit(full[FEATURES], full["label"], sample_weight=full["weight"])

    importance = final.get_booster().get_score(importance_type="gain")
    meta = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")[:-4] + "Z",
        "algorithm": "xgboost",
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": {
            "kaggle": int((data["source"] == "kaggle").sum()),
            "postgres_labelled": int((data["source"] == "postgres").sum()),
            "keyword_prior_copies": int((full["source"] == "prior").sum()),
            "fraud_rate": round(float(real["label"].mean()), 4),
        },
        "test_metrics": test_metrics,
        "feature_importance_gain": {k: round(v, 3) for k, v in importance.items()},
        "settings": {
            "amount_scale": config.AMOUNT_SCALE,
            "keyword_prior": config.KEYWORD_PRIOR,
            "label_weight": config.LABEL_WEIGHT,
        },
        "warnings": warnings,
    }
    return final, meta


def save(model: XGBClassifier, meta: dict) -> str:
    """Write models/<version>/ and point models/CURRENT at it."""
    vdir = os.path.join(config.MODEL_DIR, meta["version"])
    os.makedirs(vdir, exist_ok=True)
    model.save_model(os.path.join(vdir, "model.json"))
    with open(os.path.join(vdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    tmp = os.path.join(config.MODEL_DIR, "CURRENT.tmp")
    with open(tmp, "w") as fh:
        fh.write(meta["version"])
    os.replace(tmp, os.path.join(config.MODEL_DIR, "CURRENT"))
    return vdir
