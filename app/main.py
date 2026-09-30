"""Fraud scoring API called by the n8n "ML Fraud Scoring" node.

POST /predict   score one claim, returns the fields the Postgres node inserts
POST /retrain   retrain on Kaggle + confirmed claims, hot-swap if not worse
GET  /health    liveness + live model version and metrics
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from . import config, training
from .features import FEATURES, claim_to_features, matched_keywords

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ml-api")

app = FastAPI(title="Claims Fraud ML API", version="1.0")


class ModelHolder:
    """Keeps the live model in memory. Swapping is a single reference change under a lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._retrain_lock = threading.Lock()
        self.model: xgb.XGBClassifier | None = None
        self.meta: dict = {}

    def set(self, model, meta):
        with self._lock:
            self.model, self.meta = model, meta

    def get(self):
        with self._lock:
            return self.model, self.meta

    def load_current(self) -> bool:
        pointer = os.path.join(config.MODEL_DIR, "CURRENT")
        if not os.path.exists(pointer):
            return False
        version = open(pointer).read().strip()
        vdir = os.path.join(config.MODEL_DIR, version)
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(vdir, "model.json"))
        with open(os.path.join(vdir, "meta.json")) as fh:
            meta = json.load(fh)
        self.set(model, meta)
        log.info("Loaded model %s", version)
        return True


holder = ModelHolder()


@app.on_event("startup")
def startup():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    if holder.load_current():
        return
    log.info("No saved model, training the first one")
    try:
        model, meta = training.train(use_db=True)
        training.save(model, meta)
        holder.set(model, meta)
        log.info("Trained initial model %s: %s", meta["version"], meta["test_metrics"])
    except Exception as exc:
        log.error("Initial training failed: %s. /predict returns 503 until /retrain succeeds.", exc)


# ---------- predict ----------

class Claim(BaseModel):
    """Same keys the Edit Fields node produces under body."""
    model_config = ConfigDict(extra="allow")
    claimId: str | None = None
    policyId: str | None = None
    claimAmount: Any = None
    claimType: str | None = None
    claimDescription: str | None = None


def _risk(score: float) -> str:
    if score < config.MEDIUM_THRESHOLD:
        return "LOW"
    if score < config.HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def _flags(feats: dict, contribs: np.ndarray, description: str) -> str:
    """Readable reasons from per-feature SHAP contributions (log-odds pushed toward fraud)."""
    hits = matched_keywords(description)
    labels = {
        "claim_amount": f"High claim amount ({feats['claim_amount']:,.0f})",
        "is_major": "Major claim",
        "kw_explicit_fraud": "Suspicious description: " + ", ".join(hits["kw_explicit_fraud"]),
        "kw_duplicate": "Possible duplicate claim: " + ", ".join(hits["kw_duplicate"]),
        "kw_no_evidence": "No supporting evidence mentioned",
    }
    reasons = [
        (contribs[i], labels[name])
        for i, name in enumerate(FEATURES)
        if contribs[i] > 0.05 and (name == "claim_amount" or feats[name])
    ]
    reasons.sort(reverse=True)
    return ", ".join(r for _, r in reasons) if reasons else "No suspicious indicators"


@app.post("/predict")
def predict(claim: Claim):
    model, meta = holder.get()
    if model is None:
        raise HTTPException(503, "Model not trained yet. Add data/insurance_claims.csv and call POST /retrain.")

    feats = claim_to_features(claim.claimAmount, claim.claimType, claim.claimDescription)
    X = pd.DataFrame([feats], columns=FEATURES)
    prob = float(model.predict_proba(X)[0, 1])
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]  # drop bias

    score = int(round(prob * 100))
    risk = _risk(score)
    out = claim.model_dump()  # keeps claimId, policyId, claimType, claimDescription and any extra keys
    out.update({
        "claimAmount": feats["claim_amount"],
        "fraud_score": score,
        "risk_level": risk,
        "fraud_flags": _flags(feats, contribs, claim.claimDescription or ""),
        "status": "UNDER REVIEW" if risk == "HIGH" else "PROCESSING",
        "fraud_probability": round(prob, 4),
        "model_version": meta.get("version"),
    })
    return out


# ---------- retrain ----------

class RetrainRequest(BaseModel):
    force: bool = False
    use_db: bool = True


@app.post("/retrain")
def retrain(req: RetrainRequest | None = None, x_api_key: str | None = Header(default=None)):
    if config.RETRAIN_API_KEY and x_api_key != config.RETRAIN_API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")
    req = req or RetrainRequest()
    if not holder._retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "A retrain is already running")
    try:
        try:
            model, meta = training.train(use_db=req.use_db)
        except Exception as exc:
            raise HTTPException(500, f"Training failed: {exc}")

        _, current = holder.get()
        old_auc = current.get("test_metrics", {}).get("roc_auc")
        new_auc = meta["test_metrics"].get("roc_auc")
        if (not req.force and old_auc is not None and new_auc is not None
                and new_auc < old_auc - config.MAX_AUC_DROP):
            return {
                "promoted": False,
                "reason": f"New ROC-AUC {new_auc} is more than {config.MAX_AUC_DROP} below live {old_auc}. "
                          "Send force=true to promote anyway.",
                "live_version": current.get("version"),
                "candidate": meta,
            }

        training.save(model, meta)
        holder.set(model, meta)
        log.info("Promoted model %s: %s", meta["version"], meta["test_metrics"])
        return {"promoted": True, "previous_version": current.get("version"), "model": meta}
    finally:
        holder._retrain_lock.release()


@app.get("/health")
def health():
    model, meta = holder.get()
    return {
        "status": "ok" if model is not None else "no_model",
        "model_version": meta.get("version"),
        "test_metrics": meta.get("test_metrics"),
        "rows": meta.get("rows"),
        "warnings": meta.get("warnings"),
    }
