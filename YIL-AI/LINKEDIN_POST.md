# Publication LinkedIn prête à personnaliser

J'ai transformé plus de 76 000 rapports de maintenance aéronautique en un projet
de Machine Learning complet. ✈️🤖

Le problème : les rapports techniques contiennent beaucoup de texte libre. Leur
classement manuel par système aéronautique peut être long et répétitif.

Ma solution, **YIL AI — Aviation Maintenance Intelligence**, lit une description de défaut et propose les trois
chapitres JASC les plus plausibles — par exemple train d'atterrissage, fuselage ou
huile moteur.

AI-Powered Maintenance Report Classification.

Quelques résultats sur un test temporel indépendant 2024–2025 :

- 76 728 rapports FAA SDR B737-800 nettoyés ;
- 94,12 % d'accuracy ;
- 69,24 % de macro F1 ;
- 98,15 % de top-3 accuracy.

J'ai volontairement séparé les années : entraînement sur 2015–2022, choix du
modèle sur 2023, puis test final sur 2024–2025. Le code JASC a aussi été masqué
dans le texte pour éviter que le modèle ne « lise la réponse ».

Le modèle retenu combine TF-IDF et LinearSVC. L'application Streamlit affiche le
top-3 et signale les textes ambigus pour revue humaine. Le score n'est pas une
probabilité et le prototype ne prend aucune décision de navigabilité.

Pourquoi ce projet m'intéresse : il relie traitement du langage, qualité des
données et contraintes réelles d'un domaine critique. Une adaptation au Maroc
devra être validée sur des données locales et multilingues avec des experts
maintenance.

Données : FAA Service Difficulty Reports
https://www.faa.gov/av-info/download_SDR

Référentiel : FAA JASC
https://www.faa.gov/sites/faa.gov/files/JASC_Code.pdf

Je serais ravie d'échanger sur le NLP appliqué à la maintenance aéronautique et
sur les prochaines étapes de ce projet.

#MachineLearning #DataScience #Aviation #Maintenance #NLP #Python #Streamlit #YILAI #Morocco
