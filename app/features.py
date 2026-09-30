"""Feature engineering shared by training and prediction.

The model sees five features built from the three fields on the claim form:

    claim_amount       numeric amount
    is_major           1 if claimType is "major"
    kw_explicit_fraud  description mentions staged / fake / fraud / fabricated / false
    kw_duplicate       description mentions duplicate / already claimed
    kw_no_evidence     description mentions "no evidence"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

FEATURES = ["claim_amount", "is_major", "kw_explicit_fraud", "kw_duplicate", "kw_no_evidence"]

# Every feature can only push the score up (XGBoost monotone constraints).
# This keeps the model sensible on a small dataset, e.g. a bigger amount never lowers risk.
MONOTONE = "(" + ",".join("1" for _ in FEATURES) + ")"

KEYWORD_GROUPS = {
    "kw_explicit_fraud": ["staged", "fake", "fraud", "fabricated", "false"],
    "kw_duplicate": ["duplicate", "already claimed"],
    "kw_no_evidence": ["no evidence"],
}

MAJOR_TYPES = {"major"}
KAGGLE_MAJOR_SEVERITY = {"Major Damage", "Total Loss"}


def _to_amount(value) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def matched_keywords(description: str) -> dict[str, list[str]]:
    text = str(description or "").lower()
    return {grp: [k for k in words if k in text] for grp, words in KEYWORD_GROUPS.items()}


def claim_to_features(amount, claim_type, description) -> dict:
    """One claim (form fields or a Postgres row) -> feature dict."""
    hits = matched_keywords(description)
    row = {
        "claim_amount": _to_amount(amount),
        "is_major": int(str(claim_type or "").strip().lower() in MAJOR_TYPES),
    }
    for grp, words in hits.items():
        row[grp] = int(bool(words))
    return row


def frame_from_claims(rows: list[dict]) -> pd.DataFrame:
    """rows with claim_amount, claim_type, claim_description, is_fraud -> training frame."""
    recs = []
    for r in rows:
        f = claim_to_features(r["claim_amount"], r["claim_type"], r.get("claim_description"))
        f["label"] = int(bool(r["is_fraud"]))
        recs.append(f)
    df = pd.DataFrame(recs, columns=FEATURES + ["label"])
    df["source"] = "postgres"
    df["weight"] = config.LABEL_WEIGHT
    return df


def frame_from_kaggle(path: str) -> pd.DataFrame:
    """Map the Kaggle insurance_claims.csv onto the form's fields.

    total_claim_amount      -> claim_amount (times AMOUNT_SCALE)
    incident_severity       -> is_major (Major Damage / Total Loss)
    police_report_available == NO and witnesses == 0
                            -> kw_no_evidence (proxy, the dataset has no free text)
    fraud_reported == Y     -> label
    """
    raw = pd.read_csv(path)
    needed = {"total_claim_amount", "incident_severity", "fraud_reported"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}; is this the Kaggle insurance_claims.csv?")

    df = pd.DataFrame()
    df["claim_amount"] = pd.to_numeric(raw["total_claim_amount"], errors="coerce").fillna(0) * config.AMOUNT_SCALE
    df["is_major"] = raw["incident_severity"].isin(KAGGLE_MAJOR_SEVERITY).astype(int)
    df["kw_explicit_fraud"] = 0
    df["kw_duplicate"] = 0
    if {"police_report_available", "witnesses"} <= set(raw.columns):
        df["kw_no_evidence"] = (
            (raw["police_report_available"].astype(str).str.upper() == "NO")
            & (pd.to_numeric(raw["witnesses"], errors="coerce").fillna(1) == 0)
        ).astype(int)
    else:
        df["kw_no_evidence"] = 0
    df["label"] = (raw["fraud_reported"].astype(str).str.upper().str.strip() == "Y").astype(int)
    df["source"] = "kaggle"
    df["weight"] = 1.0
    return df


def add_keyword_prior(df: pd.DataFrame, rng: np.random.Generator,
                      frac_fraud: float = 0.20, frac_legit: float = 0.02) -> pd.DataFrame:
    """Seed the two keyword features the Kaggle data cannot teach.

    For each of kw_explicit_fraud and kw_duplicate, copy 20% of fraud rows and 2% of
    legitimate rows with that flag set to 1. This encodes the assumption that such
    wording is about 10x more common in fraudulent claims. Copies are tagged
    source="prior" and never used for evaluation. Confirmed claims from Postgres
    gradually replace this assumption with real evidence.
    """
    base = df[df["source"] == "kaggle"]
    extra = []
    for col in ("kw_explicit_fraud", "kw_duplicate"):
        for label, frac in ((1, frac_fraud), (0, frac_legit)):
            pool = base[base["label"] == label]
            n = max(1, int(round(len(pool) * frac))) if len(pool) else 0
            if n:
                idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
                copy = pool.loc[idx].copy()
                copy[col] = 1
                copy["source"] = "prior"
                extra.append(copy)
    return pd.concat([df] + extra, ignore_index=True) if extra else df
