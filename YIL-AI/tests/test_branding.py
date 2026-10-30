"""Brand-only checks; the existing business tests remain unchanged."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app_services import classify_report, result_copy_text, result_pdf
from inference import load_model_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_yil_ai_is_visible_in_the_existing_home_layout():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert sum(title.value == "YIL AI" for title in app.title) >= 2
    assert any("AVIATION MAINTENANCE INTELLIGENCE" in caption.value for caption in app.caption)


def test_copy_and_pdf_exports_use_yil_ai_brand():
    bundle = load_model_bundle(ROOT / "model" / "best_jasc_chapter_model.joblib")
    result = classify_report(bundle, "MAIN LANDING GEAR TIRE FAILED DURING TAKEOFF AND WAS REPLACED.", 0.323)
    assert result_copy_text(result).startswith("YIL AI ·")
    assert b"YIL AI" in result_pdf(result)
