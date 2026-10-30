"""Validated application operations around the unchanged JASC inference pipeline."""

from __future__ import annotations

import csv
import io
import re
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inference import JASC_CHAPTERS, normalize_text, predict_top_k


ROOT = Path(__file__).resolve().parent
MAX_REPORT_CHARS = 8000
MAX_BATCH_BYTES = 3 * 1024 * 1024
MAX_BATCH_ROWS = 200


def validate_report(text: str) -> str:
    """Apply bounded, explainable validation before calling the existing model."""
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError("Ajoutez une description de maintenance avant de lancer l'analyse.")
    if len(cleaned) > MAX_REPORT_CHARS:
        raise ValueError(f"Le rapport dépasse la limite de {MAX_REPORT_CHARS:,} caractères.")
    if len(cleaned.split()) < 3:
        raise ValueError("Décrivez l'anomalie avec au moins trois mots.")
    if len(re.findall(r"[A-Za-z]", cleaned)) < 6:
        raise ValueError("Ajoutez une description textuelle en anglais technique.")
    return cleaned


def classify_report(bundle: dict[str, Any], text: str, margin_threshold: float) -> dict[str, Any]:
    cleaned = validate_report(text)
    result = predict_top_k(bundle, cleaned, k=3)
    result["source_text"] = text
    result["margin_threshold"] = float(margin_threshold)
    result["model_name"] = str(bundle.get("selected_model", "LinearSVC"))
    return result


def explain_terms(bundle: dict[str, Any], result: dict[str, Any], limit: int = 8) -> list[tuple[str, float]]:
    """Return positive TF-IDF × coefficient terms for the leading model class."""
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    leading = result["predictions"][0]["chapter"]
    classes = [str(code).zfill(2) for code in classifier.classes_]
    if leading not in classes:
        return []
    row = vectorizer.transform([result["normalized_text"]]).tocsr()
    indices = row.indices
    contributions = row.data * classifier.coef_[classes.index(leading), indices]
    positive = np.flatnonzero(contributions > 0)
    ordered = positive[np.argsort(contributions[positive])[::-1]][:limit]
    terms = vectorizer.get_feature_names_out()
    return [(str(terms[indices[position]]), float(contributions[position])) for position in ordered]


def result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rank": item["rank"],
            "jasc_chapter": item["chapter"],
            "chapter_title": item["title"],
            "decision_score": item["decision_score"],
        }
        for item in result["predictions"]
    ]


def result_csv(result: dict[str, Any]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["report_text", "model_name", "rank", "jasc_chapter", "chapter_title", "decision_score"])
    writer.writeheader()
    writer.writerows({"report_text": result["source_text"], "model_name": result["model_name"], **row} for row in result_rows(result))
    return output.getvalue().encode("utf-8-sig")


def result_copy_text(result: dict[str, Any]) -> str:
    suggestions = "; ".join(
        f"{item['rank']}. JASC {item['chapter']} — {item['title']}" for item in result["predictions"]
    )
    return f"YIL AI · {suggestions}\nAide au classement documentaire ; vérification humaine requise."


def result_pdf(result: dict[str, Any]) -> bytes:
    """Build a local, printable report; never imply operational approval."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=46,
        bottomMargin=46,
        title="YIL AI - JASC Classification Report",
        author="YIL AI",
    )
    styles = getSampleStyleSheet()
    story = [Paragraph("YIL AI — Rapport de classement JASC", styles["Title"]), Spacer(1, 14)]
    story.append(Paragraph("Description analysée", styles["Heading2"]))
    story.append(Paragraph(escape(result["source_text"]).replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 16))
    story.append(Paragraph("Suggestions du modèle", styles["Heading2"]))
    table = Table([["Rang", "JASC", "Chapitre", "Score de décision"]] + [
        [str(row["rank"]), row["jasc_chapter"], row["chapter_title"], f"{row['decision_score']:.3f}"]
        for row in result_rows(result)
    ], colWidths=[45, 50, 250, 105])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14263B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F5FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CFDAE5")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([table, Spacer(1, 12)])
    story.append(Paragraph(f"Modèle : {escape(result['model_name'])} · Scores LinearSVC non probabilistes.", styles["BodyText"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph("Prototype d'aide au classement documentaire. Vérification humaine requise. Ce résultat ne constitue ni un diagnostic, ni une décision de navigabilité.", styles["BodyText"]))
    doc.build(story)
    return output.getvalue()


def read_batch_csv(data: bytes) -> pd.DataFrame:
    if not data:
        raise ValueError("Le fichier CSV est vide.")
    if len(data) > MAX_BATCH_BYTES:
        raise ValueError("Le fichier dépasse la limite de 3 Mo.")
    try:
        decoded = data.decode("utf-8-sig")
        header = decoded.splitlines()[0]
        delimiter = ";" if header.count(";") > header.count(",") else ","
        frame = pd.read_csv(io.StringIO(decoded), sep=delimiter, dtype=str, keep_default_na=False)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, ValueError, IndexError) as exc:
        raise ValueError("CSV illisible : utilisez un fichier UTF-8 avec une ligne d'en-têtes.") from exc
    if frame.empty or not len(frame.columns):
        raise ValueError("Le CSV doit contenir au moins une ligne et une colonne de texte.")
    if len(frame) > MAX_BATCH_ROWS:
        raise ValueError(f"Le traitement par lot est limité à {MAX_BATCH_ROWS} lignes.")
    if len(frame.columns) > 30:
        raise ValueError("Le CSV contient trop de colonnes (maximum 30).")
    if frame.columns.duplicated().any():
        raise ValueError("Les noms de colonnes doivent être uniques.")
    return frame


def classify_batch(frame: pd.DataFrame, text_column: str, bundle: dict[str, Any], margin_threshold: float) -> pd.DataFrame:
    if text_column not in frame.columns:
        raise ValueError("La colonne de texte sélectionnée est introuvable.")
    if len(frame) > MAX_BATCH_ROWS:
        raise ValueError(f"Le traitement par lot est limité à {MAX_BATCH_ROWS} lignes.")
    records = []
    for position, raw in enumerate(frame[text_column], start=1):
        try:
            result = classify_report(bundle, str(raw), margin_threshold)
            record = {"row": position, "report_text": str(raw), "status": "Classé", "error": ""}
            for item in result["predictions"]:
                record[f"jasc_{item['rank']}"] = item["chapter"]
                record[f"title_{item['rank']}"] = item["title"]
            records.append(record)
        except ValueError as exc:
            records.append({"row": position, "report_text": str(raw), "status": "À corriger", "error": str(exc)})
    return pd.DataFrame.from_records(records)


def batch_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def model_chapters(bundle: dict[str, Any]) -> pd.DataFrame:
    classes = sorted(str(code).zfill(2) for code in bundle["classifier"].classes_)
    return pd.DataFrame({"JASC": classes, "Chapitre": [JASC_CHAPTERS.get(code, f"JASC {code}") for code in classes]})
