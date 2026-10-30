# YIL AI — Aviation Maintenance Intelligence

**AI-Powered Maintenance Report Classification**

Comment exploiter le texte d’un rapport de maintenance pour identifier rapidement le système concerné ? C’est la question à laquelle répond YIL AI, mon mini-projet de machine learning appliqué à l’aéronautique.

Le principe est simple : l’utilisateur saisit une description technique en anglais, puis l’application propose les trois chapitres JASC les plus pertinents. Ces chapitres sont des catégories qui permettent de classer les rapports par système : train d’atterrissage, fuselage, huile moteur, etc. Par exemple, un rapport décrivant un problème de pneu du train d’atterrissage peut être associé au chapitre JASC 32.

Le projet repose sur 76 728 rapports nettoyés issus des données publiques FAA SDR, couvrant la période 2015–2025 et le Boeing 737-800. Après la préparation des textes, trois approches ont été comparées : un modèle de référence naïf, une régression logistique et un LinearSVC. Le modèle retenu associe une représentation TF-IDF, qui transforme les mots en valeurs numériques, à un classificateur LinearSVC.

Pour mesurer ses performances, le découpage respecte le temps : entraînement sur 2015–2022, sélection du modèle sur 2023, puis réentraînement sur 2015–2023 et test sur 2024–2025. Sur ce test de 22 263 rapports, le modèle classe correctement 94,12 % des rapports en premier choix. Le chapitre attendu figure parmi les trois suggestions dans 98,15 % des cas. Le macro F1 de 69,24 % rappelle toutefois que les performances ne sont pas homogènes entre les catégories.

Une interface Streamlit permet de tester des exemples, de consulter les résultats et d’explorer la méthode. Le projet fonctionne en local, sans SAP ni service d’IA externe pour effectuer les prédictions.

YIL AI reste un prototype d’aide au classement documentaire : il ne prédit pas les pannes et ne prend aucune décision de navigabilité. Les suggestions doivent être vérifiées par une personne compétente. Une adaptation au Maroc nécessiterait des rapports locaux autorisés et anonymisés ainsi qu’une nouvelle évaluation avec des professionnels du domaine.

Ce projet illustre une chaîne complète de machine learning : préparation des données, traitement du langage, comparaison de modèles, évaluation et création d’une application de démonstration.
