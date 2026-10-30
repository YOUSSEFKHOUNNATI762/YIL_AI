# Rapport final de validation — YIL AI

**Identité :** YIL AI · Aviation Maintenance Intelligence · AI-Powered Maintenance Report Classification.  
**Date :** 29 octobre 2026.  
**Périmètre :** rebranding de l'application documentaire existante, sans nouveau modèle ni nouvelle fonctionnalité.

## Modifications effectuées

- Interface Streamlit : nom dans la sidebar, l'accueil, le titre de l'onglet, le résumé copiable, le footer et les messages internes. La navigation, les quatre pages et la palette navy/graphite/bleu/cyan sont conservées.
- Exports : fichiers `yil_ai_classification.csv`, `yil_ai_classification.pdf` et `yil_ai_lot.csv` ; titre et auteur du PDF de classification mis à jour. La structure des résultats et le contenu métier des exports restent identiques.
- Présentation : `README.md`, `LINKEDIN_POST.md`, `PRESENTATION_PROJET.md`, licence, commentaires/docstrings de marque, couverture et synthèse PDF renommés. Le repository recommandé est `YIL-AI` ; le dossier local n'a pas été renommé pour ne pas casser les chemins.
- Captures de l'application exécutée : `assets/yil_ai_overview.jpg`, `assets/yil_ai_classification.jpg`, `assets/yil_ai_analytics.jpg`.
- Les anciens visuels et l'ancien PDF de synthèse sont conservés, récupérables dans `../outputs/yil_ai_rebrand_legacy_20261029`, hors du dossier actif livré.
- L'archive de livraison `../outputs/YIL_AI_Professional.zip` contient un dossier `YIL-AI/` sans caches ni fichiers temporaires.

## Intégrité Machine Learning

L'artefact `model/best_jasc_chapter_model.joblib`, le vectoriseur TF-IDF embarqué, les données et les rapports de métriques ne sont pas modifiés. Empreinte SHA-256 du modèle après rebranding : `3f60b682b0e832fa9500919892580ff1c7749f15bfb045504155e49699e62709`, identique au manifeste d'entraînement. Dans `inference.py`, seule la docstring de nom de projet a changé ; le code de prédiction est inchangé.

Les résultats historiques sont donc préservés : 22 263 rapports de test 2024–2025, top-1 accuracy 94,12 %, top-3 accuracy 98,15 % et macro F1 69,24 %. Ce ne sont pas des indicateurs de production en temps réel.

## Vérifications réellement exécutées

| Vérification | Résultat |
|---|---|
| Tests automatisés `python -m pytest -q --tb=short` | **25 réussis** (23 tests existants + 2 tests de marque). |
| Non-régression sur trois rapports fixes | Même ordre des trois chapitres JASC que le pipeline initial. |
| Navigation dans le navigateur local | Accueil, Classificateur, Analytique et Méthode affichés. |
| Classification réelle dans l'interface | Exemple train d'atterrissage : JASC 32, puis 53 et 25. |
| Export CSV/PDF | Génération vérifiée par les tests ; boutons visibles dans l'interface, nom et métadonnées PDF inspectés. |
| Synthèse `output/pdf/YIL_AI_Synthese.pdf` | Deux pages A4 rendues et inspectées ; titre et auteur PDF vérifiés. |
| Serveur Streamlit local | `http://localhost:8507/` répond. |
| Vérification d'environnement | Compilation Python réussie ; `pip check` sans dépendance cassée. |

## Limites conservées

YIL AI classe des textes FAA SDR Boeing 737-800 en anglais. Il ne prédit pas de pannes, ne pose aucun diagnostic et ne décide jamais de la navigabilité. Les scores LinearSVC ne sont pas des probabilités. Une adaptation opérationnelle à d'autres flottes ou au Maroc demanderait une nouvelle validation sur des données autorisées.

## Lancement

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py --server.port 8507
```
