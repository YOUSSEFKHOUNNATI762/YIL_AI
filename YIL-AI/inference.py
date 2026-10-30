"""Fonctions d'inférence pour YIL AI."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "model" / "best_jasc_chapter_model.joblib"
DEFAULT_RELIABILITY_PATH = PROJECT_DIR / "model" / "reliability_config.json"

# Intitulés officiels de niveau chapitre, adaptés en français pour l'interface.
JASC_CHAPTERS = {
    "11": "Plaques et marquages",
    "12": "Entretien courant",
    "14": "Matériel standard",
    "18": "Vibrations hélicoptère",
    "21": "Conditionnement d'air",
    "22": "Pilotage automatique",
    "23": "Communications",
    "24": "Énergie électrique",
    "25": "Équipements et aménagements",
    "26": "Protection incendie",
    "27": "Commandes de vol",
    "28": "Carburant",
    "29": "Puissance hydraulique",
    "30": "Protection glace et pluie",
    "31": "Instruments",
    "32": "Train d'atterrissage",
    "33": "Éclairage",
    "34": "Navigation",
    "35": "Oxygène",
    "36": "Pneumatique",
    "37": "Dépression",
    "38": "Eau et déchets",
    "45": "Système central de maintenance",
    "49": "Groupe auxiliaire de puissance",
    "51": "Pratiques standard et structures",
    "52": "Portes",
    "53": "Fuselage",
    "54": "Nacelles et pylônes",
    "55": "Stabilisateurs",
    "56": "Fenêtres",
    "57": "Ailes",
    "61": "Hélices et propulseurs",
    "62": "Rotor principal",
    "63": "Transmission rotor principal",
    "64": "Rotor arrière",
    "65": "Transmission rotor arrière",
    "67": "Commandes de vol rotor",
    "71": "Groupe motopropulseur",
    "72": "Turbomoteur / turbopropulseur",
    "73": "Carburant et commande moteur",
    "74": "Allumage",
    "75": "Air moteur",
    "76": "Commandes moteur",
    "77": "Indications moteur",
    "78": "Échappement moteur",
    "79": "Huile moteur",
    "80": "Démarrage",
    "81": "Suralimentation",
    "82": "Injection d'eau",
    "83": "Boîtes d'accessoires",
    "85": "Moteur à pistons",
}


def normalize_text(text: str) -> str:
    """Normalise sans supprimer le vocabulaire technique utile au modèle."""
    normalized = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def load_model_bundle(path: Path | str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    required = {"vectorizer", "classifier", "classes", "selected_model"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Modèle incomplet : clés absentes {sorted(missing)}")
    return bundle


def load_reliability_config(
    path: Path | str = DEFAULT_RELIABILITY_PATH,
) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_top_k(
    bundle: dict[str, Any],
    text: str,
    k: int = 3,
) -> dict[str, Any]:
    """Retourne les k chapitres les mieux classés et l'écart top-1/top-2.

    Les valeurs renvoyées par LinearSVC sont des scores de décision. Elles ne
    doivent pas être interprétées comme des probabilités.
    """
    cleaned = normalize_text(text)
    if len(cleaned.split()) < 3:
        raise ValueError("Décrivez l'anomalie avec au moins trois mots.")

    matrix = bundle["vectorizer"].transform([cleaned])
    classifier = bundle["classifier"]
    scores = np.asarray(classifier.decision_function(matrix)).reshape(-1)
    classes = np.asarray(classifier.classes_).astype(str)
    k = max(1, min(int(k), len(classes)))
    order = np.argsort(scores)[-k:][::-1]

    predictions = []
    for rank, index in enumerate(order, start=1):
        chapter = classes[index].zfill(2)
        predictions.append(
            {
                "rank": rank,
                "chapter": chapter,
                "title": JASC_CHAPTERS.get(chapter, f"Chapitre JASC {chapter}"),
                "decision_score": float(scores[index]),
            }
        )

    margin = (
        predictions[0]["decision_score"] - predictions[1]["decision_score"]
        if len(predictions) > 1
        else float("inf")
    )
    return {
        "normalized_text": cleaned,
        "predictions": predictions,
        "margin_top1_top2": float(margin),
        "score_type": "decision_function",
    }
