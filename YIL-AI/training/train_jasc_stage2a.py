"""Étape 2A : classification temporelle des chapitres JASC pour les B737-800.

Le script compare trois modèles sur la validation 2023, choisit le meilleur
selon le macro F1, le réentraîne sur 2015-2023, puis évalue une seule fois le
test final 2024-2025. La cible est le chapitre JASC à deux chiffres.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_INPUT = (
    WORKSPACE_DIR / "outputs" / "faa_cleaning" / "faa_sdr_b737800_jasc_ml.csv.gz"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "training_outputs"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-features", type=int, default=80_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def make_vectorizer(max_features: int) -> TfidfVectorizer:
    """TF-IDF de mots et groupes de deux mots, ajusté uniquement sur le train."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def candidate_models() -> dict[str, Any]:
    return {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        ),
        "logistic_regression_balanced": LogisticRegression(
            solver="saga",
            C=3.0,
            max_iter=350,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            tol=1e-3,
        ),
        "linear_svc_balanced": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=10_000,
        ),
    }


def decision_scores(model: Any, matrix: Any) -> tuple[np.ndarray, np.ndarray, str]:
    classes = np.asarray(model.classes_, dtype=str)
    if isinstance(model, DummyClassifier):
        # Le top-3 du modèle naïf suit les fréquences apprises, pas un ordre
        # arbitraire entre les nombreuses probabilités nulles.
        scores = np.tile(np.asarray(model.class_prior_, dtype=float), (matrix.shape[0], 1))
        return classes, scores, "class_prior"
    if hasattr(model, "predict_proba"):
        return classes, np.asarray(model.predict_proba(matrix)), "probability"
    return classes, np.asarray(model.decision_function(matrix)), "decision_function"


def top_k_details(
    y_true: pd.Series,
    classes: np.ndarray,
    scores: np.ndarray,
    k: int = 3,
) -> tuple[float, np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    order = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    top_labels = classes[order]
    top_scores = np.take_along_axis(scores, order, axis=1)
    truth = y_true.astype(str).to_numpy()
    hits = np.array([truth[index] in top_labels[index] for index in range(len(truth))])
    return float(hits.mean()), top_labels, top_scores


def evaluate_model(
    model: Any,
    matrix: Any,
    y_true: pd.Series,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, str]:
    predictions = np.asarray(model.predict(matrix), dtype=str)
    truth = y_true.astype(str).to_numpy()
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        truth,
        predictions,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        truth,
        predictions,
        average="weighted",
        zero_division=0,
    )
    classes, scores, score_type = decision_scores(model, matrix)
    top3, top_labels, top_scores = top_k_details(y_true, classes, scores, k=3)
    metrics = {
        "accuracy": accuracy_score(truth, predictions),
        "balanced_accuracy": balanced_accuracy_score(truth, predictions),
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "top3_accuracy": top3,
        "score_type": score_type,
        "evaluation_rows": len(truth),
        "evaluation_classes": int(pd.Series(truth).nunique()),
        "labels_unknown_to_model": int((~pd.Series(truth).isin(classes)).sum()),
    }
    return metrics, predictions, top_labels, top_scores, score_type


def fit_with_warnings(model: Any, matrix: Any, target: pd.Series) -> tuple[Any, list[str], float]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(matrix, target)
    elapsed = time.perf_counter() - started
    warning_messages = sorted({str(item.message) for item in captured})
    return model, warning_messages, elapsed


def per_class_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    training_target: pd.Series,
) -> pd.DataFrame:
    labels = sorted(set(y_true.astype(str)) | set(map(str, predictions)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true.astype(str),
        predictions,
        labels=labels,
        zero_division=0,
    )
    training_counts = training_target.astype(str).value_counts()
    result = pd.DataFrame(
        {
            "jasc_chapter": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "test_support": support.astype(int),
            "train_validation_support": [int(training_counts.get(label, 0)) for label in labels],
        }
    )
    return result.sort_values(["test_support", "jasc_chapter"], ascending=[False, True])


def top_confusions(y_true: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    pairs = pd.DataFrame(
        {"true_jasc_chapter": y_true.astype(str).to_numpy(), "predicted_jasc_chapter": predictions}
    )
    pairs = pairs.loc[pairs["true_jasc_chapter"] != pairs["predicted_jasc_chapter"]]
    if pairs.empty:
        return pd.DataFrame(columns=["true_jasc_chapter", "predicted_jasc_chapter", "count"])
    return (
        pairs.value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["count", "true_jasc_chapter", "predicted_jasc_chapter"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def save_confusion_plot(
    y_true: pd.Series,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    labels = sorted(set(y_true.astype(str)) | set(map(str, predictions)))
    matrix = confusion_matrix(y_true.astype(str), predictions, labels=labels, normalize="true")
    figure, axis = plt.subplots(figsize=(17, 15))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Part des rapports")
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Chapitre JASC prédit",
        ylabel="Chapitre JASC réel",
        title="Matrice de confusion normalisée — test 2024-2025",
    )
    plt.setp(axis.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def dataframe_predictions(
    test: pd.DataFrame,
    predictions: np.ndarray,
    top_labels: np.ndarray,
    top_scores: np.ndarray,
    score_type: str,
) -> pd.DataFrame:
    result = test[
        [
            "record_id",
            "source_year",
            "difficulty_date",
            "aircraft_group_id",
            "jasc_chapter",
            "discrepancy_model_text",
        ]
    ].copy()
    result = result.rename(columns={"jasc_chapter": "true_jasc_chapter"})
    result["predicted_jasc_chapter"] = predictions
    result["correct"] = result["true_jasc_chapter"].astype(str).to_numpy() == predictions
    for index in range(top_labels.shape[1]):
        result[f"top_{index + 1}_chapter"] = top_labels[:, index]
        result[f"top_{index + 1}_score"] = top_scores[:, index]
    result["score_type"] = score_type
    result["top_1_minus_top_2_score"] = top_scores[:, 0] - top_scores[:, 1]
    return result


def model_card_text(
    best_name: str,
    validation_row: dict[str, Any],
    test_row: dict[str, Any],
    vectorizer_features: int,
    input_path: Path,
) -> str:
    return f"""# Modèle de classification des chapitres JASC — Étape 2A

## But

Le modèle lit le texte nettoyé d'un rapport FAA concernant un B737-800 et propose un chapitre JASC à deux chiffres. Il aide au classement documentaire. Il ne prédit pas une défaillance future, une RUL, une décision de navigabilité ou une autorisation de vol.

## Données

- Source préparée : `{input_path.name}`
- Entraînement initial : 2015-2022
- Sélection du modèle : validation 2023
- Test final : 2024-2025
- Variable d'entrée : `discrepancy_model_text`
- Cible : `jasc_chapter`
- Nombre de variables TF-IDF finales : {vectorizer_features:,}

Le texte destiné au modèle masque toute occurrence exacte du code JASC cible. Les champs post-diagnostic, l'identifiant du rapport et l'exploitant ne sont pas utilisés comme variables.

## Modèle sélectionné

- Algorithme : `{best_name}`
- Critère de sélection : macro F1 sur la validation 2023
- Macro F1 validation : {validation_row['macro_f1']:.4f}
- Macro F1 test : {test_row['macro_f1']:.4f}
- Accuracy test : {test_row['accuracy']:.4f}
- Top-3 accuracy test : {test_row['top3_accuracy']:.4f}

## Utilisation

Le fichier `best_jasc_chapter_model.joblib` contient le vectoriseur TF-IDF, le classifieur, la liste des classes et les métadonnées. Une prédiction doit utiliser le même nettoyage de texte que l'étape 1.

## Limites

- Les rapports FAA sont des déclarations de difficultés en service, pas des séries temporelles continues de capteurs.
- Le modèle est entraîné sur les libellés présents dans les données historiques américaines.
- Les classes rares restent moins fiables que les classes fréquentes.
- Les résultats ne remplacent ni l'AMM, ni la MEL, ni un diagnostic d'ingénierie, ni l'approbation d'une autorité ou d'un organisme agréé.
- Une utilisation marocaine exige une validation locale avec des rapports, une terminologie et des procédures représentatifs des opérateurs marocains.
"""


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Base nettoyée introuvable : {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "record_id",
        "source_year",
        "difficulty_date",
        "dataset_split",
        "aircraft_group_id",
        "discrepancy_model_text",
        "jasc_chapter",
    ]
    data = pd.read_csv(args.input, usecols=usecols, dtype="string")
    if data[usecols].isna().any().any():
        missing = data[usecols].isna().sum()
        raise ValueError(f"Valeurs manquantes dans les colonnes requises :\n{missing[missing > 0]}")
    if not data["record_id"].is_unique:
        raise ValueError("record_id doit être unique avant l'entraînement.")
    if not data["jasc_chapter"].str.fullmatch(r"\d{2}").all():
        raise ValueError("Chaque chapitre JASC doit contenir exactement deux chiffres.")

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(data["dataset_split"].unique())
    if actual_splits != expected_splits:
        raise ValueError(f"Partitions inattendues : {actual_splits}")

    train = data.loc[data["dataset_split"].eq("train")].reset_index(drop=True)
    validation = data.loc[data["dataset_split"].eq("validation")].reset_index(drop=True)
    test = data.loc[data["dataset_split"].eq("test")].reset_index(drop=True)

    train_labels = set(train["jasc_chapter"])
    validation_unseen = sorted(set(validation["jasc_chapter"]) - train_labels)
    if validation_unseen:
        raise ValueError(f"La validation contient des chapitres absents du train : {validation_unseen}")

    vectorizer = make_vectorizer(args.max_features)
    vectorize_started = time.perf_counter()
    x_train = vectorizer.fit_transform(train["discrepancy_model_text"])
    x_validation = vectorizer.transform(validation["discrepancy_model_text"])
    vectorize_seconds = time.perf_counter() - vectorize_started

    comparison_rows: list[dict[str, Any]] = []
    fitted_candidates: dict[str, Any] = {}
    validation_details: dict[str, dict[str, Any]] = {}
    for name, estimator in candidate_models().items():
        fitted, warning_messages, fit_seconds = fit_with_warnings(
            estimator,
            x_train,
            train["jasc_chapter"],
        )
        metrics, predictions, top_labels, top_scores, score_type = evaluate_model(
            fitted,
            x_validation,
            validation["jasc_chapter"],
        )
        row = {
            "phase": "validation_2023",
            "candidate": name,
            "selected": False,
            "train_rows": len(train),
            "evaluation_rows": len(validation),
            "tfidf_features": x_train.shape[1],
            "fit_seconds": fit_seconds,
            "warnings_count": len(warning_messages),
            "warnings": " | ".join(warning_messages),
            **metrics,
        }
        comparison_rows.append(row)
        fitted_candidates[name] = fitted
        validation_details[name] = {
            "metrics": metrics,
            "predictions": predictions,
            "top_labels": top_labels,
            "top_scores": top_scores,
            "score_type": score_type,
        }

    best_name = max(
        comparison_rows,
        key=lambda item: (item["macro_f1"], item["weighted_f1"], item["accuracy"]),
    )["candidate"]
    for row in comparison_rows:
        row["selected"] = row["candidate"] == best_name
    selected_validation_row = next(row for row in comparison_rows if row["selected"])

    # Après sélection sur 2023, réentraînement final sur 2015-2023.
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_vectorizer = make_vectorizer(args.max_features)
    final_vectorize_started = time.perf_counter()
    x_train_validation = final_vectorizer.fit_transform(
        train_validation["discrepancy_model_text"]
    )
    x_test = final_vectorizer.transform(test["discrepancy_model_text"])
    final_vectorize_seconds = time.perf_counter() - final_vectorize_started

    final_estimator = clone(candidate_models()[best_name])
    final_estimator, final_warnings, final_fit_seconds = fit_with_warnings(
        final_estimator,
        x_train_validation,
        train_validation["jasc_chapter"],
    )
    test_metrics, test_predictions, test_top_labels, test_top_scores, score_type = evaluate_model(
        final_estimator,
        x_test,
        test["jasc_chapter"],
    )
    test_row = {
        "phase": "test_2024_2025",
        "candidate": best_name,
        "selected": True,
        "train_rows": len(train_validation),
        "evaluation_rows": len(test),
        "tfidf_features": x_train_validation.shape[1],
        "fit_seconds": final_fit_seconds,
        "warnings_count": len(final_warnings),
        "warnings": " | ".join(final_warnings),
        **test_metrics,
    }

    comparison = pd.DataFrame(comparison_rows + [test_row])
    per_class = per_class_metrics(
        test["jasc_chapter"],
        test_predictions,
        train_validation["jasc_chapter"],
    )
    confusions = top_confusions(test["jasc_chapter"], test_predictions)
    predictions = dataframe_predictions(
        test,
        test_predictions,
        test_top_labels,
        test_top_scores,
        score_type,
    )

    paths = {
        "comparison": args.output_dir / "model_comparison.csv",
        "per_class": args.output_dir / "test_metrics_per_class.csv",
        "predictions": args.output_dir / "test_predictions.csv.gz",
        "confusions": args.output_dir / "test_top_confusions.csv",
        "plot": args.output_dir / "test_confusion_matrix_normalized.png",
        "model": args.output_dir / "best_jasc_chapter_model.joblib",
        "model_card": args.output_dir / "model_card.md",
        "report": args.output_dir / "run_report.json",
    }

    comparison.to_csv(paths["comparison"], index=False, encoding="utf-8-sig")
    per_class.to_csv(paths["per_class"], index=False, encoding="utf-8-sig")
    predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8",
        compression={"method": "gzip", "compresslevel": 5, "mtime": 0},
    )
    confusions.to_csv(paths["confusions"], index=False, encoding="utf-8-sig")
    save_confusion_plot(test["jasc_chapter"], test_predictions, paths["plot"])

    model_bundle = {
        "vectorizer": final_vectorizer,
        "classifier": final_estimator,
        "target": "jasc_chapter",
        "text_column": "discrepancy_model_text",
        "classes": [str(value) for value in final_estimator.classes_],
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "trained_through_year": 2023,
        "test_years": [2024, 2025],
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(model_bundle, paths["model"], compress=3)
    paths["model_card"].write_text(
        model_card_text(
            best_name,
            selected_validation_row,
            test_row,
            x_train_validation.shape[1],
            args.input,
        ),
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Étape 2A uniquement : classification des chapitres JASC B737-800",
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "rows": len(data),
        },
        "split": {
            "train_2015_2022": len(train),
            "validation_2023": len(validation),
            "test_2024_2025": len(test),
        },
        "method": {
            "selection_metric": "macro_f1",
            "selected_model": best_name,
            "random_state": RANDOM_STATE,
            "tfidf": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "max_df": 0.995,
                "max_features": args.max_features,
                "final_features": x_train_validation.shape[1],
            },
            "vectorize_train_validation_seconds": vectorize_seconds,
            "vectorize_final_seconds": final_vectorize_seconds,
        },
        "validation_results": comparison_rows,
        "test_result": test_row,
        "quality_checks": {
            "unique_record_ids": bool(data["record_id"].is_unique),
            "missing_required_values": int(data[usecols].isna().sum().sum()),
            "validation_unseen_chapters": validation_unseen,
            "test_unseen_chapters_vs_train_validation": sorted(
                set(test["jasc_chapter"]) - set(train_validation["jasc_chapter"])
            ),
            "test_prediction_rows": len(predictions),
            "prediction_ids_unique": bool(predictions["record_id"].is_unique),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    # Contrôles bloquants avant le rapport final.
    assert len(predictions) == len(test)
    assert predictions["record_id"].is_unique
    assert predictions["predicted_jasc_chapter"].str.fullmatch(r"\d{2}").all()
    assert comparison.loc[comparison["phase"].eq("validation_2023"), "selected"].sum() == 1
    assert test_row["labels_unknown_to_model"] <= 1

    report["outputs"] = {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
        if name != "report"
    }
    paths["report"].write_text(
        json.dumps(json_value(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "selected_model": best_name,
                "validation_macro_f1": selected_validation_row["macro_f1"],
                "test_macro_f1": test_row["macro_f1"],
                "test_accuracy": test_row["accuracy"],
                "test_top3_accuracy": test_row["top3_accuracy"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
