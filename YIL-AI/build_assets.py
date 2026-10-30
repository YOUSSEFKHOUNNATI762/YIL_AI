"""Génère les visuels du portfolio LinkedIn."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
REPORTS = ROOT / "reports"
BLUE = "#0078D4"
NAVY = "#0B1F33"
CYAN = "#31C3E7"
PALE = "#EAF4FB"
GOLD = "#F4B942"


def rounded_box(axis, xy, width, height, color, title, subtitle=""):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0,
        facecolor=color,
    )
    axis.add_patch(box)
    axis.text(xy[0] + width / 2, xy[1] + height * 0.62, title, ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    if subtitle:
        axis.text(xy[0] + width / 2, xy[1] + height * 0.30, subtitle, ha="center", va="center", fontsize=9, color="white")


def cover() -> None:
    figure, axis = plt.subplots(figsize=(12, 6.27), dpi=100)
    gradient = np.linspace(0, 1, 800)
    gradient = np.vstack([gradient] * 500)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("aviation", [NAVY, "#0B4778", BLUE])
    axis.imshow(gradient, extent=[0, 12, 0, 6.27], origin="lower", cmap=cmap, aspect="auto")
    axis.axis("off")
    axis.text(0.7, 5.35, "YIL AI", color="white", fontsize=34, fontweight="bold")
    axis.text(0.7, 4.78, "AI-Powered Maintenance Report Classification", color="#DCEEFF", fontsize=19)
    axis.text(0.7, 4.35, "FAA SDR · B737-800 · NLP · JASC", color=CYAN, fontsize=14, fontweight="bold")

    metrics = [
        ("76 728", "rapports nettoyés"),
        ("94,1 %", "accuracy test"),
        ("98,1 %", "top-3 test"),
    ]
    for index, (value, label) in enumerate(metrics):
        x = 0.7 + index * 2.55
        rounded_box(axis, (x, 2.6), 2.15, 1.12, "#116AA4", value, label)

    axis.text(1.25, 1.50, "Rapport texte", color="white", fontsize=13, fontweight="bold", ha="center")
    axis.annotate("", xy=(2.85, 1.50), xytext=(2.05, 1.50), arrowprops={"arrowstyle": "->", "color": CYAN, "lw": 2})
    axis.text(3.55, 1.50, "TF-IDF", color="white", fontsize=13, fontweight="bold", ha="center")
    axis.annotate("", xy=(5.15, 1.50), xytext=(4.35, 1.50), arrowprops={"arrowstyle": "->", "color": CYAN, "lw": 2})
    axis.text(6.05, 1.50, "LinearSVC", color="white", fontsize=13, fontweight="bold", ha="center")
    axis.annotate("", xy=(7.65, 1.50), xytext=(6.95, 1.50), arrowprops={"arrowstyle": "->", "color": CYAN, "lw": 2})
    axis.text(8.85, 1.50, "Top-3 JASC", color="white", fontsize=13, fontweight="bold", ha="center")
    axis.text(0.7, 0.55, "Test temporel indépendant : 2024–2025", color="#DCEEFF", fontsize=12)
    axis.text(11.35, 0.55, "Portfolio ML", color=GOLD, fontsize=12, fontweight="bold", ha="right")
    figure.savefig(ASSETS / "linkedin_cover.png", bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def model_comparison() -> None:
    data = pd.read_csv(REPORTS / "model_comparison.csv")
    validation = data.loc[data["phase"] == "validation_2023"].copy()
    names = {
        "dummy_most_frequent": "Naïf",
        "logistic_regression_balanced": "Régression\nlogistique",
        "linear_svc_balanced": "LinearSVC",
    }
    validation["label"] = validation["candidate"].map(names)
    x = np.arange(len(validation))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5.8), dpi=160)
    figure.patch.set_facecolor("white")
    colors = ["#8EC5E8", BLUE, NAVY]
    for offset, metric, label in [
        (-width, "accuracy", "Accuracy"),
        (0, "macro_f1", "Macro F1"),
        (width, "top3_accuracy", "Top-3"),
    ]:
        bars = axis.bar(x + offset, validation[metric] * 100, width, label=label, color=colors[["accuracy", "macro_f1", "top3_accuracy"].index(metric)])
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_title("Comparaison des modèles — validation 2023", loc="left", fontsize=17, fontweight="bold", color=NAVY)
    axis.set_ylabel("Score (%)")
    axis.set_xticks(x, validation["label"])
    axis.set_ylim(0, 106)
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.text(0, -0.18, "Sélection sur le macro F1 · aucune donnée 2024–2025 utilisée pour choisir le modèle", transform=axis.transAxes, fontsize=9, color="#4B6073")
    figure.tight_layout()
    figure.savefig(ASSETS / "model_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def workflow() -> None:
    figure, axis = plt.subplots(figsize=(12, 4), dpi=160)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")
    steps = [
        ("1", "FAA SDR", "2015–2025"),
        ("2", "Nettoyage", "anti-fuite de cible"),
        ("3", "TF-IDF", "mots + bigrammes"),
        ("4", "LinearSVC", "classes équilibrées"),
        ("5", "Top-3 JASC", "+ revue humaine"),
    ]
    colors = [NAVY, "#124A73", BLUE, "#1385B5", "#159AB7"]
    for index, (number, title, subtitle) in enumerate(steps):
        x = 0.3 + index * 2.35
        rounded_box(axis, (x, 1.3), 1.85, 1.45, colors[index], f"{number}. {title}", subtitle)
        if index < len(steps) - 1:
            axis.annotate("", xy=(x + 2.25, 2.02), xytext=(x + 1.88, 2.02), arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 2.5})
    axis.set_title("Du rapport brut à une aide au classement", loc="left", fontsize=18, fontweight="bold", color=NAVY)
    figure.tight_layout()
    figure.savefig(ASSETS / "workflow.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    cover()
    model_comparison()
    workflow()
    print("Visuels générés dans", ASSETS)
