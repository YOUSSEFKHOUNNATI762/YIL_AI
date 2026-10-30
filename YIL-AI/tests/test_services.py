"""Pure validation, export, batch, explainability, and model regression tests."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from app_services import (
    MAX_BATCH_ROWS,
    classify_batch,
    classify_report,
    explain_terms,
    model_chapters,
    read_batch_csv,
    result_copy_text,
    result_csv,
    result_pdf,
    validate_report,
)
from inference import load_model_bundle, predict_top_k


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle(ROOT / "model" / "best_jasc_chapter_model.joblib")


def test_artifact_hash_matches_training_manifest():
    manifest = json.loads((ROOT / "reports" / "run_report.json").read_text(encoding="utf-8"))
    expected = manifest["outputs"]["model"]["sha256"]
    actual = hashlib.sha256((ROOT / "model" / "best_jasc_chapter_model.joblib").read_bytes()).hexdigest()
    assert actual == expected


def test_original_pipeline_predictions_preserved(bundle):
    originals = list(csv.DictReader((ROOT / "data" / "sample_reports.csv").open(encoding="utf-8-sig")))
    baseline = json.loads((ROOT / "tests" / "fixtures" / "baseline_predictions.json").read_text(encoding="utf-8"))
    for source, expected in zip(originals, baseline, strict=True):
        assert source["example_id"] == expected["example_id"]
        original = predict_top_k(bundle, source["report_text"])
        updated = classify_report(bundle, source["report_text"], 0.3238222803105735)
        observed = [item["chapter"] for item in updated["predictions"]]
        assert observed == expected["chapters"]
        assert len(observed) == len(set(observed)) == 3
        assert updated["predictions"] == original["predictions"]


@pytest.mark.parametrize("text", ["", "OIL", "123 456 789", "x " * 4500])
def test_invalid_reports_rejected(text):
    with pytest.raises(ValueError):
        validate_report(text)


def test_explanation_uses_real_tfidf_coefficients(bundle):
    result = classify_report(bundle, "MAIN LANDING GEAR TIRE FAILED DURING TAKEOFF AND WAS REPLACED.", 0.323)
    terms = explain_terms(bundle, result)
    assert terms and len(terms) <= 8
    assert all(contribution > 0 for _, contribution in terms)
    assert all(term in bundle["vectorizer"].vocabulary_ for term, _ in terms)


def test_jasc_reference_matches_model_classes(bundle):
    chapters = model_chapters(bundle)
    assert len(chapters) == len(bundle["classifier"].classes_)
    assert chapters["JASC"].is_unique
    assert chapters.loc[chapters["JASC"] == "32", "Chapitre"].iloc[0] == "Train d'atterrissage"


def test_single_exports_contain_real_result(bundle):
    result = classify_report(bundle, "MAIN LANDING GEAR TIRE FAILED DURING TAKEOFF AND WAS REPLACED.", 0.323)
    exported = list(csv.DictReader(io.StringIO(result_csv(result).decode("utf-8-sig"))))
    assert [row["jasc_chapter"] for row in exported] == [p["chapter"] for p in result["predictions"]]
    assert all(row["report_text"] == result["source_text"] for row in exported)
    assert all(row["model_name"] == "linear_svc_balanced" for row in exported)
    assert "JASC 32" in result_copy_text(result)
    pdf = result_pdf(result)
    assert pdf.startswith(b"%PDF-") and len(pdf) > 1000


def test_batch_csv_valid_invalid_and_export(bundle):
    raw = ("report_text\n"
           '"MAIN LANDING GEAR TIRE FAILED DURING TAKEOFF."\n'
           '"OIL"\n').encode("utf-8")
    frame = read_batch_csv(raw)
    output = classify_batch(frame, "report_text", bundle, 0.323)
    assert output["status"].tolist() == ["Classé", "À corriger"]
    assert output.iloc[0]["jasc_1"] == "32"
    assert "À corriger" in output.to_csv(index=False)
    assert len(output) == 2


def test_semicolon_separated_csv_is_accepted():
    frame = read_batch_csv(b"report_text;reference\nLANDING GEAR TIRE FAILED;EX-1\n")
    assert frame.columns.tolist() == ["report_text", "reference"]
    assert frame.iloc[0]["reference"] == "EX-1"


@pytest.mark.parametrize("content", [b"", b"\xff\xfe\xfd", b"report_text\n", b"x" * (3 * 1024 * 1024 + 1)], ids=["empty", "binary", "header_only", "oversize"])
def test_invalid_csv_rejected(content):
    with pytest.raises(ValueError):
        read_batch_csv(content)


def test_batch_limit_and_column_checked(bundle):
    oversized = pd.DataFrame({"report": ["THREE VALID WORDS"] * (MAX_BATCH_ROWS + 1)})
    with pytest.raises(ValueError):
        classify_batch(oversized, "report", bundle, 0.323)
    with pytest.raises(ValueError):
        classify_batch(oversized.head(1), "missing", bundle, 0.323)
