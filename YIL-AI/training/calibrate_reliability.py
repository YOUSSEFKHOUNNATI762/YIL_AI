"""Calibre sur 2023 un seuil d'écart top-1/top-2 pour la démonstration."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "faa_cleaning" / "faa_sdr_b737800_jasc_ml.csv.gz"
FINAL_PREDICTIONS = ROOT / "outputs" / "faa_jasc_stage2a" / "test_predictions.csv.gz"
FINAL_MODEL = PROJECT / "model" / "best_jasc_chapter_model.joblib"
OUTPUT = PROJECT / "model" / "reliability_config.json"
TARGET_ACCEPTED_ACCURACY = 0.95


def select_threshold(margins: np.ndarray, correct: np.ndarray) -> tuple[float, int]:
    order = np.argsort(margins)[::-1]
    sorted_correct = correct[order].astype(float)
    cumulative_accuracy = np.cumsum(sorted_correct) / np.arange(1, len(correct) + 1)
    eligible = np.flatnonzero(cumulative_accuracy >= TARGET_ACCEPTED_ACCURACY)
    if len(eligible) == 0:
        best = int(np.argmax(cumulative_accuracy))
    else:
        best = int(eligible[-1])
    accepted_count = best + 1
    threshold = float(margins[order[best]])
    return threshold, accepted_count


def main() -> None:
    data = pd.read_csv(
        DATA_PATH,
        usecols=["dataset_split", "discrepancy_model_text", "jasc_chapter"],
    )
    train = data.loc[data["dataset_split"] == "train"].copy()
    validation = data.loc[data["dataset_split"] == "validation"].copy()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=80_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train["discrepancy_model_text"])
    x_validation = vectorizer.transform(validation["discrepancy_model_text"])
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=10_000,
    ).fit(x_train, train["jasc_chapter"].astype(str))

    scores = np.asarray(model.decision_function(x_validation))
    order = np.argsort(scores, axis=1)[:, ::-1]
    predictions = np.asarray(model.classes_).astype(str)[order[:, 0]]
    margins = scores[np.arange(len(scores)), order[:, 0]] - scores[
        np.arange(len(scores)), order[:, 1]
    ]
    correct = predictions == validation["jasc_chapter"].astype(str).to_numpy()
    threshold, accepted_count = select_threshold(margins, correct)
    accepted = margins >= threshold

    test = pd.read_csv(FINAL_PREDICTIONS)
    test_margin = test["top_1_minus_top_2_score"].to_numpy()
    test_accepted = test_margin >= threshold

    bundle = joblib.load(FINAL_MODEL)
    config = {
        "method": "top1_minus_top2_linear_svc_decision_score",
        "margin_threshold": threshold,
        "target_validation_accuracy_for_accepted": TARGET_ACCEPTED_ACCURACY,
        "validation_year": 2023,
        "validation_rows": int(len(validation)),
        "validation_accepted_rows": int(accepted_count),
        "validation_coverage": float(accepted.mean()),
        "validation_accuracy_accepted": float(correct[accepted].mean()),
        "validation_accuracy_review": float(correct[~accepted].mean()),
        "test_years": [2024, 2025],
        "test_rows": int(len(test)),
        "test_coverage_at_frozen_threshold": float(test_accepted.mean()),
        "test_accuracy_accepted_audit": float(test.loc[test_accepted, "correct"].mean()),
        "test_accuracy_review_audit": float(test.loc[~test_accepted, "correct"].mean()),
        "model_selected": bundle["selected_model"],
        "warning": (
            "Seuil heuristique calibre sur la validation 2023 avec un modele entraine "
            "sur 2015-2022. Le score LinearSVC n'est pas une probabilite."
        ),
    }
    OUTPUT.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
