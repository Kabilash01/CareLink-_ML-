"""
CARELINK AI — Model Training Script
Uses GradientBoostingClassifier (sklearn 1.4.x compatible, no version mismatch on load).
Run:  python training/train_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_CSV     = ROOT / "training" / "structured_training_dataset.csv"
FALLBACK_CSV = ROOT / "data"    / "structured_training_dataset.csv"
MODEL_OUT    = ROOT / "app"     / "models" / "risk_model.pkl"
META_OUT     = ROOT / "app"     / "models" / "model_metadata.json"

FEATURE_COLS = [
    "age",
    "duration_days",
    "num_high_symptoms",
    "num_medium_symptoms",
    "num_low_symptoms",
    "chronic_count",
    "high_weight",
    "medium_weight",
    "chronic_weight",
    "age_factor",
    "duration_factor",
    "is_elderly",
    "is_child",
    "has_critical_symptom",
]
LABEL_MAP   = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LABEL_NAMES = ["LOW", "MEDIUM", "HIGH"]


def load_data() -> pd.DataFrame:
    for path in (DATA_CSV, FALLBACK_CSV):
        if path.exists():
            print(f"Loading data from {path}")
            return pd.read_csv(path)
    sys.exit(f"ERROR: Training CSV not found at {DATA_CSV} or {FALLBACK_CSV}.\n"
             f"Run:  python data/synthetic_generator.py  first.")


def build_features(df: pd.DataFrame):
    # Only keep columns that exist (old CSV has 6; new CSV has 14+)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).values
    y = df["risk_level"].map(LABEL_MAP).values
    print(f"Features used ({len(available)}): {available}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def main():
    df = load_data()
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── pipeline: scaler + GradientBoosting ───────────────────────────
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gbc", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=10,
            random_state=42,
        )),
    ])

    # ── cross-validation ──────────────────────────────────────────────
    print("\nRunning 5-fold stratified CV …")
    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro", n_jobs=-1
    )
    print(f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── final fit ─────────────────────────────────────────────────────
    clf.fit(X_train, y_train)

    # ── evaluation ────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    print("\nClassification Report (test set):")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    acc     = (y_pred == y_test).mean()
    f1_mac  = cross_val_score(clf, X_test, y_test, scoring="f1_macro", cv=3).mean()
    print(f"\nTest Accuracy : {acc:.4f}")

    # ── save ──────────────────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"\nModel saved → {MODEL_OUT}")

    meta = {
        "version":             "v2.0.0",
        "model_type":          "GradientBoostingClassifier",
        "sklearn_version":     __import__("sklearn").__version__,
        "features":            [c for c in FEATURE_COLS if c in df.columns],
        "label_map":           LABEL_MAP,
        "cv_f1_macro":         round(float(cv_scores.mean()), 4),
        "test_accuracy":       round(float(acc), 4),
        "n_estimators":        200,
        "training_samples":    int(len(X_train)),
        "class_distribution":  {LABEL_NAMES[k]: int(v)
                                 for k, v in zip(*np.unique(y, return_counts=True))},
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved → {META_OUT}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
