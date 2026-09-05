"""
Train the root-cause classifier and prove it beats a code-only lookup.

Two systems, same train/test split:
  A) rule_lookup : error_code -> majority cause      (the dumb baseline)
  B) tfidf_lr    : free-text description + context   (the trained model)

If (B) does not clearly beat (A), the model has no right to exist and you
should say so out loud. That comparison IS your justification -- a panel
will ask "why not just a dict?" and this table is the answer.

Usage:
    python -m models.train_classifier --events artifacts/events.jsonl
"""
import argparse
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

from data.taxonomy import CAUSES, CODE_LOOKUP

TEXT_COL = "error_description"
CAT_COLS = ["error_code", "error_source", "error_step", "error_reason", "method"]
NUM_COLS = ["attempt_number", "amount_rupees"]


def load(path):
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame([{
        "error_description": r["error_description"],
        "error_code": r["error_code"],
        "error_source": r["error_source"],
        "error_step": r["error_step"],
        "error_reason": r["error_reason"],
        "method": r["method"],
        "attempt_number": r["attempt_number"],
        "amount_rupees": r["amount_paise"] / 100,
        "y": r["ground_truth_cause"],
        "event_id": r["event_id"],
    } for r in rows])
    return df


def build_pipeline():
    pre = ColumnTransformer([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True), TEXT_COL),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True), TEXT_COL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ], remainder="drop")
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced")),
    ])


def eval_block(name, y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"accuracy  {acc:.4f}")
    print(f"macro-F1  {mf1:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0,
                                digits=3))
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(mf1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="artifacts/events.jsonl")
    ap.add_argument("--model-out", default="artifacts/cause_clf.joblib")
    ap.add_argument("--metrics-out", default="artifacts/classifier_metrics.json")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = load(a.events)
    tr, te = train_test_split(df, test_size=a.test_size, random_state=a.seed,
                              stratify=df["y"])
    print(f"train {len(tr)}   test {len(te)}")

    # ---- A) lookup baseline -------------------------------------------------
    base_pred = te["error_code"].map(CODE_LOOKUP).fillna(CAUSES[0])
    m_base = eval_block("rule_lookup (error_code only)", te["y"], base_pred, CAUSES)

    # ---- B) trained model ---------------------------------------------------
    pipe = build_pipeline()
    pipe.fit(tr, tr["y"])
    model_pred = pipe.predict(te)
    m_model = eval_block("tfidf_lr (description + context)", te["y"],
                         model_pred, CAUSES)

    print("\nconfusion matrix (rows=true, cols=pred), order:")
    print("  " + "  ".join(f"{c[:9]:>9s}" for c in CAUSES))
    cm = confusion_matrix(te["y"], model_pred, labels=CAUSES)
    for c, row in zip(CAUSES, cm):
        print(f"{c:22s} " + " ".join(f"{v:9d}" for v in row))

    lift = m_model["macro_f1"] - m_base["macro_f1"]
    print(f"\nmacro-F1 lift over lookup: +{lift:.4f}")
    print("NOTE: labels carry 5% injected noise, so ~0.95 is the practical ceiling.")

    joblib.dump(pipe, a.model_out)

    # Per-event predictions, used later by the experiment so the agent runs on
    # PREDICTED causes -- never on ground truth. Leaking truth here would
    # invalidate every rupee number downstream.
    pred_map = dict(zip(te["event_id"], model_pred))
    with open("artifacts/test_predictions.json", "w") as f:
        json.dump(pred_map, f)

    with open(a.metrics_out, "w") as f:
        json.dump({"rule_lookup": m_base, "tfidf_lr": m_model,
                   "macro_f1_lift": round(float(lift), 4),
                   "n_train": len(tr), "n_test": len(te)}, f, indent=2)
    print(f"\nsaved model -> {a.model_out}")
    print(f"saved metrics -> {a.metrics_out}")
    print(f"saved {len(pred_map)} test predictions -> artifacts/test_predictions.json")


if __name__ == "__main__":
    main()
