# Modèle de classification des chapitres JASC — Étape 2A

## But

Le modèle lit le texte nettoyé d'un rapport FAA concernant un B737-800 et propose un chapitre JASC à deux chiffres. Il aide au classement documentaire. Il ne prédit pas une défaillance future, une RUL, une décision de navigabilité ou une autorisation de vol.

## Données

- Source préparée : `faa_sdr_b737800_jasc_ml.csv.gz`
- Entraînement initial : 2015-2022
- Sélection du modèle : validation 2023
- Test final : 2024-2025
- Variable d'entrée : `discrepancy_model_text`
- Cible : `jasc_chapter`
- Nombre de variables TF-IDF finales : 80,000

Le texte destiné au modèle masque toute occurrence exacte du code JASC cible. Les champs post-diagnostic, l'identifiant du rapport et l'exploitant ne sont pas utilisés comme variables.

## Modèle sélectionné

- Algorithme : `linear_svc_balanced`
- Critère de sélection : macro F1 sur la validation 2023
- Macro F1 validation : 0.8135
- Macro F1 test : 0.6924
- Accuracy test : 0.9412
- Top-3 accuracy test : 0.9815

## Utilisation

Le fichier `best_jasc_chapter_model.joblib` contient le vectoriseur TF-IDF, le classifieur, la liste des classes et les métadonnées. Une prédiction doit utiliser le même nettoyage de texte que l'étape 1.

## Limites

- Les rapports FAA sont des déclarations de difficultés en service, pas des séries temporelles continues de capteurs.
- Le modèle est entraîné sur les libellés présents dans les données historiques américaines.
- Les classes rares restent moins fiables que les classes fréquentes.
- Les résultats ne remplacent ni l'AMM, ni la MEL, ni un diagnostic d'ingénierie, ni l'approbation d'une autorité ou d'un organisme agréé.
- Une utilisation marocaine exige une validation locale avec des rapports, une terminologie et des procédures représentatifs des opérateurs marocains.
