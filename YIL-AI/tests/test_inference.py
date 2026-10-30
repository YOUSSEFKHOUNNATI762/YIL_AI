from pathlib import Path

from inference import load_model_bundle, normalize_text, predict_top_k


PROJECT = Path(__file__).resolve().parents[1]


def test_normalize_text_collapses_spaces():
    assert normalize_text("  OIL\n\tLEAK   FOUND ") == "OIL LEAK FOUND"


def test_prediction_returns_sorted_top_three():
    bundle = load_model_bundle(PROJECT / "model" / "best_jasc_chapter_model.joblib")
    result = predict_top_k(
        bundle,
        "MAIN LANDING GEAR TIRE FAILED DURING TAKEOFF AND WAS REPLACED.",
        k=3,
    )
    assert len(result["predictions"]) == 3
    scores = [item["decision_score"] for item in result["predictions"]]
    assert scores == sorted(scores, reverse=True)
    assert all(len(item["chapter"]) == 2 for item in result["predictions"])
    assert result["margin_top1_top2"] >= 0


def test_too_short_report_is_rejected():
    bundle = load_model_bundle(PROJECT / "model" / "best_jasc_chapter_model.joblib")
    try:
        predict_top_k(bundle, "OIL LEAK", k=3)
    except ValueError as exc:
        assert "trois mots" in str(exc)
    else:
        raise AssertionError("Un texte trop court doit être rejeté")
