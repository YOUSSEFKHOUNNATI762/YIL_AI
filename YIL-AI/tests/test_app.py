"""User journeys for the four Streamlit workspaces."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from inference import load_model_bundle


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def warm_model():
    load_model_bundle(PROJECT / "model" / "best_jasc_chapter_model.joblib")


def app() -> AppTest:
    return AppTest.from_file(str(PROJECT / "streamlit_app.py"), default_timeout=30).run()


def test_overview_and_main_navigation():
    at = app()
    assert not at.exception
    assert any(button.label == "Classify a Report" for button in at.button)
    assert len(at.metric) == 4
    at.button(key="hero_classify").click().run()
    assert not at.exception
    assert len(at.text_area) == 1
    at.button(key="nav_Analytique").click().run()
    assert not at.exception
    assert len(at.metric) == 4
    at.button(key="nav_Méthode").click().run()
    assert not at.exception
    assert len(at.dataframe) == 1


def test_single_report_history_and_reset():
    at = app()
    at.button(key="overview_example_0").click().run()
    assert at.text_area(key="report_text").value.startswith("MAIN LANDING GEAR")
    at.button(key="classify").click().run()
    assert not at.exception
    assert at.session_state.analysis_result["predictions"][0]["chapter"] == "32"
    assert len(at.session_state.history) == 1
    assert len(at.get("download_button")) >= 2
    at.button(key="nav_Analytique").click().run()
    at.button(key="nav_Classificateur").click().run()
    assert at.text_area(key="report_text").value.startswith("MAIN LANDING GEAR")
    at.button(key="reset_report").click().run()
    assert at.text_area(key="report_text").value == ""
    assert at.session_state.analysis_result is None
    at.session_state.classifier_mode = "Historique"
    at.run()
    assert not at.exception
    assert len(at.session_state.history) == 1
    at.button(key="restore_0").click().run()
    assert at.session_state.analysis_result["predictions"][0]["chapter"] == "32"
    at.session_state.classifier_mode = "Historique"
    at.run()
    at.button(key="clear_history").click().run()
    assert at.session_state.history == []


def test_invalid_report_does_not_keep_old_result():
    at = app()
    at.button(key="overview_example_2").click().run()
    at.button(key="classify").click().run()
    assert at.session_state.analysis_result is not None
    at.text_area(key="report_text").set_value("OIL").run()
    assert at.session_state.analysis_result is None
    at.button(key="classify").click().run()
    assert at.error
    assert not at.exception


def test_batch_empty_state_and_page_switch():
    at = app()
    at.button(key="nav_Classificateur").click().run()
    at.session_state.classifier_mode = "Lot CSV"
    at.run()
    assert not at.exception
    assert at.info
    at.button(key="nav_Méthode").click().run()
    assert not at.exception
