"""Construit la fiche PDF de synthèse du projet YIL AI."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "pdf" / "YIL_AI_Synthese.pdf"
BLUE = colors.HexColor("#0078D4")
NAVY = colors.HexColor("#0B1F33")
PALE = colors.HexColor("#EAF4FB")
MID = colors.HexColor("#4B6073")
GREEN = colors.HexColor("#107C10")


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\segoeui.ttf")
    bold = Path(r"C:\Windows\Fonts\segoeuib.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("SegoeUI", regular))
        pdfmetrics.registerFont(TTFont("SegoeUI-Bold", bold))
        return "SegoeUI", "SegoeUI-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6E2EA"))
    canvas.line(20 * mm, 14 * mm, 190 * mm, 14 * mm)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(MID)
    canvas.drawString(20 * mm, 9 * mm, "YIL AI - Aviation Maintenance Intelligence")
    canvas.drawRightString(190 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "TitleCustom",
        fontName=FONT_BOLD,
        fontSize=24,
        leading=28,
        textColor=NAVY,
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "SubtitleCustom",
        fontName=FONT,
        fontSize=12,
        leading=17,
        textColor=MID,
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "H1Custom",
        fontName=FONT_BOLD,
        fontSize=16,
        leading=20,
        textColor=BLUE,
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "H2Custom",
        fontName=FONT_BOLD,
        fontSize=11,
        leading=14,
        textColor=NAVY,
        spaceBefore=2 * mm,
        spaceAfter=1 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "BodyCustom",
        fontName=FONT,
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#273746"),
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "SmallCustom",
        fontName=FONT,
        fontSize=7.8,
        leading=10.5,
        textColor=MID,
    )
)
styles.add(
    ParagraphStyle(
        "MetricValue",
        fontName=FONT_BOLD,
        fontSize=17,
        leading=19,
        textColor=BLUE,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        "MetricLabel",
        fontName=FONT,
        fontSize=7.7,
        leading=10,
        textColor=MID,
        alignment=TA_CENTER,
    )
)


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", styles["BodyCustom"])


def section_box(title: str, body: str, color=PALE) -> Table:
    content = [
        Paragraph(title, styles["H2Custom"]),
        Paragraph(body, styles["BodyCustom"]),
    ]
    table = Table([[content]], colWidths=[170 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C6D9E6")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="YIL AI - Aviation Maintenance Intelligence",
        author="YIL AI",
    )
    story = []

    story.append(Paragraph("YIL AI", styles["TitleCustom"]))
    story.append(
        Paragraph(
            "Aviation Maintenance Intelligence<br/>AI-Powered Maintenance Report Classification",
            styles["SubtitleCustom"],
        )
    )

    cover_path = ROOT / "assets" / "linkedin_cover.png"
    story.append(Image(str(cover_path), width=170 * mm, height=88.8 * mm))
    story.append(Spacer(1, 5 * mm))

    metrics = [
        ("76 728", "rapports nettoyés"),
        ("94,12 %", "accuracy test"),
        ("69,24 %", "macro F1 test"),
        ("98,15 %", "top-3 test"),
    ]
    metric_cells = [
        [Paragraph(value, styles["MetricValue"]), Paragraph(label, styles["MetricLabel"])]
        for value, label in metrics
    ]
    metric_table = Table([metric_cells], colWidths=[42.5 * mm] * 4, rowHeights=[18 * mm])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F9FC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6E2EA")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6E2EA")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 4 * mm))
    story.append(
        section_box(
            "L'essentiel",
            "Entrée : une description technique en anglais. Sortie : les trois chapitres "
            "JASC les plus plausibles, leur score de décision et un signal de vérification "
            "humaine. Il s'agit d'une aide au classement documentaire, pas d'une prédiction "
            "de panne.",
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Données et protocole", styles["H1Custom"]))
    story.extend(
        [
            bullet("Source : FAA Service Difficulty Reports, éditions annuelles 2015-2025."),
            bullet("Périmètre : 76 728 rapports B737-800 après nettoyage."),
            bullet("Cible : chapitre JASC à deux chiffres, par exemple 32 train d'atterrissage."),
            bullet("Anti-fuite : les codes JASC explicites sont masqués dans le texte."),
            bullet("Découpage temporel : train 2015-2022, validation 2023, test 2024-2025."),
        ]
    )
    story.append(Spacer(1, 2 * mm))
    workflow_path = ROOT / "assets" / "workflow.png"
    story.append(Image(str(workflow_path), width=170 * mm, height=56.7 * mm))

    story.append(Paragraph("Étape 2 expliquée simplement", styles["H1Custom"]))
    explanation_data = [
        [Paragraph("1. Transformer le texte", styles["H2Custom"]), Paragraph("TF-IDF convertit les mots et groupes de deux mots en nombres exploitables par les modèles.", styles["BodyCustom"])],
        [Paragraph("2. Comparer", styles["H2Custom"]), Paragraph("Un modèle naïf, une régression logistique et un LinearSVC sont comparés sur l'année 2023.", styles["BodyCustom"])],
        [Paragraph("3. Choisir", styles["H2Custom"]), Paragraph("Le macro F1 donne autant d'importance aux classes rares qu'aux classes fréquentes. LinearSVC est retenu.", styles["BodyCustom"])],
        [Paragraph("4. Tester une fois", styles["H2Custom"]), Paragraph("Le modèle final est évalué sur 2024-2025, qui n'a pas servi au choix du modèle.", styles["BodyCustom"])],
    ]
    explanation = Table(explanation_data, colWidths=[48 * mm, 122 * mm])
    explanation.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6E2EA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(explanation)

    story.append(Paragraph("Résultats et prudence", styles["H1Custom"]))
    story.append(
        Paragraph(
            "Le meilleur modèle atteint 94,12 % d'accuracy et 98,15 % de top-3 sur le test. "
            "Le macro F1 de 69,24 % rappelle que certaines classes rares restent difficiles. "
            "Un écart top-1/top-2 inférieur à 0,324 déclenche une recommandation de revue. "
            "Ce seuil est heuristique et le score LinearSVC n'est pas une probabilité.",
            styles["BodyCustom"],
        )
    )

    story.append(Paragraph("Prochaine évolution pour le Maroc", styles["H1Custom"]))
    next_move = Table(
        [
            [bullet("Collecter un corpus local autorisé et anonymisé."), bullet("Tester l'anglais, le français et, si utile, l'arabe.")],
            [bullet("Faire valider la taxonomie par des experts maintenance."), bullet("Mesurer les erreurs par classe et surveiller la dérive.")],
        ],
        colWidths=[85 * mm, 85 * mm],
    )
    next_move.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(next_move)
    story.append(Spacer(1, 2 * mm))
    story.append(
        section_box(
            "Limite de sécurité",
            "Ce prototype ne pose aucun diagnostic, ne prédit pas une défaillance, "
            "ne décide pas de la navigabilité et n'autorise aucune remise en service.",
            color=colors.HexColor("#FFF4CE"),
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "Sources : "
            '<link href="https://www.faa.gov/av-info/download_SDR" color="#0067B8">FAA SDR</link>'
            " - "
            '<link href="https://www.faa.gov/sites/faa.gov/files/JASC_Code.pdf" color="#0067B8">FAA JASC</link>',
            styles["SmallCustom"],
        )
    )

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
