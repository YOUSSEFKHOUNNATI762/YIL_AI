"""YIL AI — local Streamlit product interface."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from app_services import (
    MAX_BATCH_BYTES,
    MAX_BATCH_ROWS,
    MAX_REPORT_CHARS,
    batch_csv,
    classify_batch,
    classify_report,
    explain_terms,
    model_chapters,
    read_batch_csv,
    result_copy_text,
    result_csv,
    result_pdf,
)
from inference import load_model_bundle, load_reliability_config


ROOT = Path(__file__).resolve().parent
PAGES = {
    "Vue d'ensemble": ("space_dashboard", "Vue d'ensemble"),
    "Classificateur": ("manage_search", "Classificateur JASC"),
    "Analytique": ("monitoring", "Analytique du modèle"),
    "Méthode": ("menu_book", "Méthode et sources"),
}
EXAMPLES = {
    "Train d'atterrissage": "MAIN LANDING GEAR TIRE CAP LOSS DURING TAKEOFF. MAINTENANCE REPLACED THE NUMBER 2 TIRE ASSEMBLY.",
    "Structure / fuselage": "CRACK FOUND ON THE RIGHT FUSELAGE SKIN DURING SCHEDULED INSPECTION. THE DAMAGED AREA WAS REPAIRED IN ACCORDANCE WITH THE STRUCTURAL MANUAL.",
    "Huile moteur": "ENGINE OIL FILTER BYPASS LIGHT ILLUMINATED IN FLIGHT. CREW SHUT DOWN THE ENGINE AND MAINTENANCE REPLACED THE OIL INDICATOR.",
}


@st.cache_resource(show_spinner=False)
def model_bundle():
    """Load only the shipped artifact after comparing its recorded SHA-256."""
    import json

    path = ROOT / "model" / "best_jasc_chapter_model.joblib"
    manifest = json.loads((ROOT / "reports" / "run_report.json").read_text(encoding="utf-8"))
    expected = manifest["outputs"]["model"]["sha256"].lower()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("L'empreinte du modèle local ne correspond pas au rapport d'entraînement.")
    bundle = load_model_bundle(path)
    if bundle.get("selected_model") != "linear_svc_balanced":
        raise ValueError("L'artefact fourni n'est pas le modèle LinearSVC attendu.")
    return bundle


@st.cache_data(show_spinner=False)
def reliability():
    return load_reliability_config(ROOT / "model" / "reliability_config.json")


@st.cache_data(show_spinner=False)
def comparison_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "model_comparison.csv", encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def chapter_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "test_metrics_per_class.csv", encoding="utf-8-sig", dtype={"jasc_chapter": str})


@st.cache_data(show_spinner=False)
def confusion_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "test_top_confusions.csv", encoding="utf-8-sig", dtype={"true_jasc_chapter": str, "predicted_jasc_chapter": str})


def go(page: str) -> None:
    st.session_state.page = page


def load_example(name: str) -> None:
    st.session_state.page = "Classificateur"
    st.session_state.classifier_mode = "Un rapport"
    st.session_state.report_text = EXAMPLES[name]
    st.session_state.analysis_result = None


def reset_report() -> None:
    st.session_state.report_text = ""
    st.session_state.analysis_result = None


def invalidate_result() -> None:
    st.session_state.analysis_result = None


def restore_history(index: int) -> None:
    record = st.session_state.history[index]
    st.session_state.report_text = record["source_text"]
    st.session_state.analysis_result = record
    st.session_state.classifier_mode = "Un rapport"
    st.session_state.page = "Classificateur"


def safe_bundle():
    try:
        return model_bundle()
    except Exception:
        logging.exception("YIL AI model unavailable")
        st.error("Le modèle local est introuvable, modifié ou incompatible. Réinstallez les dépendances et restaurez l'artefact livré avec le projet.")
        return None


def section_label(text: str) -> None:
    st.caption(text.upper())


def metric_row(test: pd.Series) -> None:
    with st.container(horizontal=True, key="metric_row"):
        st.metric("Top-1 accuracy", f"{test['accuracy'] * 100:.2f} %".replace(".", ","), border=True)
        st.metric("Top-3 accuracy", f"{test['top3_accuracy'] * 100:.2f} %".replace(".", ","), border=True)
        st.metric("Macro F1", f"{test['macro_f1'] * 100:.2f} %".replace(".", ","), border=True)
        st.metric("Rapports de test", f"{int(test['evaluation_rows']):,}".replace(",", " "), border=True)


def overview(test: pd.Series) -> None:
    with st.container(border=True, key="hero"):
        section_label("Aviation Maintenance Intelligence · édition professionnelle")
        st.title("YIL AI")
        st.write("AI-Powered Maintenance Report Classification. Classez un rapport de maintenance aéronautique en anglais et examinez trois chapitres JASC proposés par un modèle de langage statistique local.")
        st.button("Classify a Report", key="hero_classify", icon=":material/arrow_forward:", type="primary", on_click=go, args=("Classificateur",))
        st.caption("Périmètre du modèle : rapports FAA SDR · Boeing 737-800 · 2015–2025")
    section_label("Résultats du test indépendant 2024–2025")
    metric_row(test)
    left, right = st.columns([1.55, 1], gap="medium")
    with left:
        with st.container(border=True, key="overview_workflow"):
            st.subheader("Du rapport au chapitre", icon=":material/route:")
            st.markdown("**01 · Décrire** l'anomalie en anglais technique.\n\n**02 · Classer** avec TF-IDF et LinearSVC.\n\n**03 · Vérifier** les trois suggestions et le rapport d'origine.")
            st.button("Comprendre la méthode", icon=":material/arrow_outward:", type="tertiary", on_click=go, args=("Méthode",), key="overview_method")
    with right:
        with st.container(border=True, key="overview_scope"):
            st.subheader("Ce que fait l'outil", icon=":material/fact_check:")
            st.write("Aide au tri documentaire des systèmes aéronautiques. Les résultats sont à confirmer par un professionnel.")
            st.caption("Le modèle n'effectue aucun diagnostic ni décision de navigabilité.")
    section_label("Exemples de rapports")
    with st.container(horizontal=True, key="overview_examples"):
        for index, name in enumerate(EXAMPLES):
            st.button(name, icon=":material/north_east:", key=f"overview_example_{index}", on_click=load_example, args=(name,), width="stretch")


def result_panel(result: dict, bundle: dict) -> None:
    top = result["predictions"][0]
    with st.container(border=True, key="primary_result"):
        section_label("Primary JASC suggestion · rang 1")
        st.metric("Code proposé", f"JASC {top['chapter']}", label_visibility="collapsed")
        st.subheader(top["title"])
        st.write(f"Le modèle associe ce rapport au chapitre « {top['title']} » du référentiel JASC.")
        if result["margin_top1_top2"] < result["margin_threshold"]:
            st.badge("Scores proches · vérification renforcée", color="orange", icon=":material/rule:")
        else:
            st.badge("Suggestion à confirmer", color="blue", icon=":material/person_check:")
    st.subheader("Autres chapitres à examiner")
    alternative_cols = st.columns(2, gap="small")
    for column, item in zip(alternative_cols, result["predictions"][1:]):
        with column.container(border=True, key=f"alternative_{item['rank']}"):
            section_label(f"Rang {item['rank']}")
            st.markdown(f"**JASC {item['chapter']} · {item['title']}**")
    with st.expander("Pourquoi ces mots ont compté ?", icon=":material/insights:"):
        terms = explain_terms(bundle, result)
        if terms:
            st.write("Termes avec une contribution positive au score du chapitre principal. Il s'agit de coefficients statistiques, pas d'une explication aéronautique.")
            st.dataframe(pd.DataFrame(terms, columns=["Terme TF-IDF", "Contribution"]), hide_index=True, alt="Termes du rapport ayant contribué au score du premier chapitre")
        else:
            st.caption("Aucun terme positif n'a été isolé pour cette suggestion.")
    with st.expander("Scores et texte analysé", icon=":material/tune:"):
        st.dataframe(pd.DataFrame([{"Rang": p["rank"], "JASC": p["chapter"], "Chapitre": p["title"], "Score de décision": round(p["decision_score"], 3)} for p in result["predictions"]]), hide_index=True, alt="Trois suggestions JASC et leurs scores de décision LinearSVC")
        st.caption(f"Écart des deux premiers scores : {result['margin_top1_top2']:.3f} · seuil heuristique : {result['margin_threshold']:.3f}. Ce seuil provient de la validation 2023 et peut changer d'échelle après le réentraînement final. Les scores ne sont pas des probabilités.")
        st.text(result["source_text"])
    st.caption("Copiez le résumé avec le bouton intégré au bloc ci-dessous.")
    st.code(result_copy_text(result), language=None, wrap_lines=True)
    with st.container(horizontal=True, key="export_row"):
        st.download_button("Exporter CSV", result_csv(result), file_name="yil_ai_classification.csv", mime="text/csv", icon=":material/download:", on_click="ignore")
        try:
            pdf_data = result_pdf(result)
            st.download_button("Exporter PDF", pdf_data, file_name="yil_ai_classification.pdf", mime="application/pdf", icon=":material/picture_as_pdf:", on_click="ignore")
        except Exception:
            logging.exception("YIL AI PDF export failed")
            st.warning("L'export PDF est indisponible dans cet environnement. L'export CSV reste accessible.")


def single_classifier() -> None:
    left, right = st.columns([2.15, 1], gap="medium")
    with left:
        with st.container(border=True, key="input_card"):
            st.subheader("Rapport de maintenance", icon=":material/description:")
            st.caption("Décrivez le défaut observé en anglais. Le modèle classe le texte du rapport, sans données de capteurs.")
            st.text_area("Description de l'anomalie", key="report_text", height=220, placeholder="Ex. MAIN LANDING GEAR TIRE FAILED DURING TAKEOFF…", on_change=invalidate_result, persist_state="session")
            count = len(st.session_state.report_text or "")
            st.caption(f"{count:,} / {MAX_REPORT_CHARS:,} caractères conseillés".replace(",", " "))
            with st.container(horizontal=True, key="input_actions"):
                classify = st.button("Classer le rapport", key="classify", icon=":material/arrow_forward:", type="primary")
                st.button("Réinitialiser", key="reset_report", icon=":material/restart_alt:", on_click=reset_report)
        if classify:
            st.session_state.analysis_result = None
            try:
                with st.spinner("Classement en cours avec le modèle local…"):
                    bundle = safe_bundle()
                    if bundle is not None:
                        result = classify_report(bundle, st.session_state.report_text, reliability()["margin_threshold"])
                        st.session_state.analysis_result = result
                        st.session_state.history.insert(0, result)
                        st.session_state.history = st.session_state.history[:10]
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                logging.exception("YIL AI single inference failed")
                st.error("Le classement n'a pas abouti. Vérifiez le rapport et réessayez.")
        if st.session_state.analysis_result is not None:
            bundle = safe_bundle()
            if bundle is not None:
                result_panel(st.session_state.analysis_result, bundle)
        else:
            with st.container(border=True, key="result_empty"):
                section_label("Espace des résultats")
                st.subheader("Les trois suggestions apparaîtront ici.")
                st.write("Commencez avec un rapport ou sélectionnez un exemple.")
    with right:
        with st.container(border=True, key="example_card"):
            st.subheader("Exemples", icon=":material/bookmarks:")
            for index, name in enumerate(EXAMPLES):
                st.button(name, key=f"classifier_example_{index}", icon=":material/north_east:", on_click=load_example, args=(name,), width="stretch")
        with st.container(border=True, key="scope_card"):
            section_label("Périmètre")
            st.markdown("**Source**  \nFAA Service Difficulty Reports")
            st.markdown("**Avion**  \nBoeing 737-800")
            st.markdown("**Sortie**  \nTrois chapitres JASC à deux chiffres")
            st.caption("Aide au classement documentaire · vérification humaine requise")


def batch_classifier() -> None:
    with st.container(border=True, key="batch_card"):
        st.subheader("Classer un fichier CSV", icon=":material/upload_file:")
        st.write(f"Importez un CSV UTF-8 de 3 Mo et {MAX_BATCH_ROWS} lignes au maximum. Choisissez la colonne contenant les descriptions en anglais.")
        uploaded = st.file_uploader("Fichier CSV", type="csv", key="batch_upload", on_change=lambda: st.session_state.update(batch_result=None))
        if uploaded is None:
            st.info("Aucun fichier importé. Chaque ligne valide produira trois suggestions JASC.", icon=":material/info:")
            return
        try:
            if uploaded.size > MAX_BATCH_BYTES:
                raise ValueError("Le fichier dépasse la limite de 3 Mo.")
            frame = read_batch_csv(uploaded.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return
        st.caption(f"{len(frame)} lignes · {len(frame.columns)} colonnes")
        text_column = st.selectbox("Colonne du rapport", frame.columns.tolist(), key="batch_column", on_change=lambda: st.session_state.update(batch_result=None))
        st.dataframe(frame[[text_column]].head(5), hide_index=True, alt="Aperçu des cinq premières descriptions du fichier importé")
        if st.button("Classer les lignes", key="batch_classify", type="primary", icon=":material/play_arrow:"):
            st.session_state.batch_result = None
            try:
                with st.spinner("Classement des lignes valides…"):
                    bundle = safe_bundle()
                    if bundle is not None:
                        st.session_state.batch_result = classify_batch(frame, text_column, bundle, reliability()["margin_threshold"])
            except Exception:
                logging.exception("YIL AI batch inference failed")
                st.error("Le traitement du CSV n'a pas abouti.")
        result = st.session_state.batch_result
        if result is not None:
            good = int((result["status"] == "Classé").sum())
            st.success(f"{good} lignes classées sur {len(result)} ; les lignes invalides sont indiquées dans le tableau.")
            st.dataframe(result, hide_index=True, alt="Résultats des classifications JASC par ligne du CSV")
            st.download_button("Exporter le résultat CSV", batch_csv(result), file_name="yil_ai_lot.csv", mime="text/csv", on_click="ignore", icon=":material/download:")


def history_view() -> None:
    with st.container(border=True, key="history_card"):
        st.subheader("Historique de cette session", icon=":material/history:")
        st.caption("Les dix derniers classements restent dans cet onglet jusqu'à la fermeture de la session ou au redémarrage de l'application.")
        if not st.session_state.history:
            st.info("Aucune classification dans cette session.", icon=":material/info:")
            return
        st.button("Effacer l'historique", key="clear_history", icon=":material/delete_sweep:", on_click=lambda: st.session_state.update(history=[]))
        for index, record in enumerate(st.session_state.history):
            top = record["predictions"][0]
            with st.container(border=True, key=f"history_{index}"):
                st.markdown(f"**JASC {top['chapter']} · {top['title']}**")
                st.caption(record["source_text"][:210] + ("…" if len(record["source_text"]) > 210 else ""))
                st.button("Revoir ce rapport", key=f"restore_{index}", on_click=restore_history, args=(index,), icon=":material/open_in_new:")


def classifier() -> None:
    section_label("Analyse documentaire")
    st.header("Classificateur JASC")
    st.write("Saisissez un rapport, traitez un petit lot ou consultez les résultats de cette session.")
    mode = st.segmented_control("Mode de travail", ["Un rapport", "Lot CSV", "Historique"], key="classifier_mode", persist_state="session")
    if mode == "Lot CSV":
        batch_classifier()
    elif mode == "Historique":
        history_view()
    else:
        single_classifier()


def analytics(test: pd.Series) -> None:
    section_label("Évaluation vérifiable")
    st.header("Analytique du modèle")
    st.write("Mesures historiques du test 2024–2025 ; elles ne décrivent pas une activité en temps réel.")
    metric_row(test)
    comparison = comparison_data().query("phase == 'validation_2023'").copy()
    comparison["Modèle"] = comparison["candidate"].map({"dummy_most_frequent": "Référence naïve", "logistic_regression_balanced": "Régression logistique", "linear_svc_balanced": "LinearSVC"})
    comparison["Macro F1 (%)"] = comparison["macro_f1"] * 100
    chart, notes = st.columns([1.7, 1], gap="medium")
    with chart.container(border=True, key="model_compare"):
        st.subheader("Sélection sur la validation 2023")
        st.bar_chart(comparison, x="Modèle", y="Macro F1 (%)", horizontal=True, color="#2563EB", height=250, alt="LinearSVC obtient le meilleur macro F1 sur la validation 2023")
        st.caption("La régression logistique a atteint sa limite d'itérations sans convergence complète.")
    with notes.container(border=True, key="metric_guide"):
        st.subheader("Lire les chiffres")
        st.markdown("**Accuracy :** part des premiers choix corrects.  \n**Top-3 :** part des rapports dont la classe attendue figure dans les trois suggestions.  \n**Précision :** parmi les rapports classés dans une catégorie, part des bons.  \n**Rappel :** parmi les vrais rapports d'une catégorie, part retrouvée.  \n**Macro F1 :** moyenne des F1 où chaque catégorie a le même poids.")
    chapters = chapter_data()
    col1, col2 = st.columns(2, gap="medium")
    with col1.container(border=True, key="class_perf"):
        st.subheader("F1 par chapitre")
        sample = chapters.sort_values("test_support", ascending=False).head(12).copy()
        sample["Chapitre"] = "JASC " + sample["jasc_chapter"]
        sample["F1 (%)"] = sample["f1"] * 100
        st.bar_chart(sample, x="Chapitre", y="F1 (%)", horizontal=True, color="#38BDF8", height=430, alt="F1 des douze chapitres les plus représentés dans le test")
        st.caption("Douze chapitres ayant le plus de rapports dans le jeu de test.")
    with col2.container(border=True, key="class_volume"):
        st.subheader("Répartition du test")
        sample = chapters.sort_values("test_support", ascending=False).head(12).copy()
        sample["Chapitre"] = "JASC " + sample["jasc_chapter"]
        st.bar_chart(sample, x="Chapitre", y="test_support", horizontal=True, color="#2563EB", height=430, alt="Effectifs des douze chapitres les plus représentés dans le test")
        st.caption("Les volumes proviennent du rapport de métriques par classe.")
    with st.container(border=True, key="confusions"):
        st.subheader("Confusions les plus fréquentes")
        st.write("Ces couples correspondent à des erreurs observées, et ne constituent pas une matrice de confusion complète.")
        errors = confusion_data().head(15).copy()
        errors["Couple"] = "JASC " + errors["true_jasc_chapter"] + " → " + errors["predicted_jasc_chapter"]
        st.bar_chart(errors, x="Couple", y="count", horizontal=True, color="#5A88E8", height=390, alt="Quinze couples de chapitres les plus souvent confondus")
    with st.expander("Table complète des performances par chapitre", icon=":material/table_chart:"):
        st.dataframe(chapters, hide_index=True, alt="Précision, rappel, F1 et effectifs de chaque chapitre du test")
        st.caption("Le test comporte une classe absente de l'entraînement final : le chapitre 14. Elle ne peut pas être proposée par le modèle.")


def methodology() -> None:
    section_label("Traçabilité du projet")
    st.header("Méthode et documentation")
    st.write("D'où viennent les données, comment fonctionne le modèle et où s'arrêtent ses conclusions.")
    with st.container(border=True, key="method_pipeline"):
        st.subheader("Chaîne de traitement")
        st.mermaid_chart("flowchart LR\nA[Rapport de maintenance] --> B[Préparation du texte]\nB --> C[TF-IDF]\nC --> D[LinearSVC]\nD --> E[Top 3 chapitres JASC]", alt="Rapport de maintenance, préparation du texte, TF-IDF, LinearSVC, puis trois chapitres JASC")
    left, right = st.columns([1.5, 1], gap="medium")
    with left.container(border=True, key="method_details"):
        st.subheader("Données et apprentissage")
        st.markdown("**Source.** Rapports publics FAA Service Difficulty Reports.  \n**Périmètre.** Boeing 737-800, 2015–2025, 76 728 rapports nettoyés.  \n**Entrée.** Description technique en anglais, avec masquage des codes JASC cibles dans le texte d'entraînement.  \n**Représentation.** TF-IDF sur mots et groupes de deux mots, jusqu'à 80 000 caractéristiques.  \n**Modèle.** LinearSVC avec pondération des classes, retenu selon le macro F1 de validation.")
        st.markdown("**Séparation chronologique.** Entraînement initial 2015–2022 (45 065 rapports), validation 2023 (9 400), réentraînement jusqu'à 2023, puis test final 2024–2025 (22 263).")
    with right.container(border=True, key="method_limits"):
        st.subheader("Limites à connaître")
        st.write("Les classes peu représentées sont plus difficiles. Le modèle a été entraîné sur des rapports américains en anglais et doit être réévalué avant une adaptation à d'autres flottes ou langues.")
        st.warning("Aide au classement documentaire uniquement. Aucun diagnostic, aucune prévision de panne et aucune décision de navigabilité.", icon=":material/warning:")
        st.caption("Le seuil d'écart entre les deux premiers scores reste expérimental et n'est pas une probabilité.")
    with st.container(border=True, key="reference_card"):
        st.subheader("Référence JASC présente dans le modèle")
        bundle = safe_bundle()
        if bundle is not None:
            table = model_chapters(bundle)
            search = st.text_input("Rechercher un code ou un chapitre", key="jasc_search", placeholder="Ex. 32, train, moteur")
            if search.strip():
                table = table[table.apply(lambda row: search.casefold() in " ".join(row.astype(str)).casefold(), axis=1)]
            st.dataframe(table, hide_index=True, alt="Codes et libellés des chapitres JASC effectivement prédictibles")
            st.caption(f"{len(bundle['classifier'].classes_)} chapitres prédictibles dans l'artefact livré.")
    with st.container(border=True, key="sources_card"):
        st.subheader("Sources et ressources")
        st.link_button("Données FAA SDR", "https://www.faa.gov/av-info/download_SDR", icon=":material/open_in_new:")
        st.link_button("Référentiel JASC FAA", "https://www.faa.gov/sites/faa.gov/files/JASC_Code.pdf", icon=":material/open_in_new:")
        pdf = ROOT / "output" / "pdf" / "YIL_AI_Synthese.pdf"
        if pdf.exists():
            st.download_button("Synthèse du projet", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf", icon=":material/download:", on_click="ignore")


def run() -> None:
    st.set_page_config(page_title="YIL AI — Aviation Maintenance Intelligence", page_icon=":material/flight:", layout="wide", initial_sidebar_state="expanded")
    st.html(ROOT / "assets" / "interface.css")
    for key, value in {"page": "Vue d'ensemble", "report_text": "", "analysis_result": None, "history": [], "classifier_mode": "Un rapport", "batch_result": None}.items():
        st.session_state.setdefault(key, value)
    with st.sidebar:
        with st.container(key="brand"):
            st.title("YIL AI", icon=":material/flight:")
            st.caption("AVIATION MAINTENANCE INTELLIGENCE")
        st.caption("WORKSPACE")
        for page, (icon, _) in PAGES.items():
            st.button(page, key=f"nav_{page}", icon=f":material/{icon}:", on_click=go, args=(page,), type="secondary" if st.session_state.page == page else "tertiary", width="stretch")
        with st.container(border=True, key="sidebar_status"):
            st.caption("MODÈLE LOCAL")
            st.markdown("**FAA SDR · Boeing 737-800**")
            st.caption("Classification JASC à 2 chiffres")
        st.caption("Prototype documentaire · 2026")
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", key="topbar"):
        st.markdown(f"**{PAGES[st.session_state.page][1]}**")
        st.badge("Inférence locale", color="blue", icon=":material/verified:")
    try:
        comparison = comparison_data()
        test = comparison.loc[comparison["phase"] == "test_2024_2025"].iloc[0]
    except Exception:
        logging.exception("YIL AI metrics unavailable")
        st.error("Les rapports de métriques du projet sont introuvables ou illisibles.")
        st.stop()
    page = st.session_state.page
    if page == "Vue d'ensemble":
        overview(test)
    elif page == "Classificateur":
        classifier()
    elif page == "Analytique":
        try:
            analytics(test)
        except Exception:
            logging.exception("YIL AI analytics unavailable")
            st.error("Certaines données d'évaluation sont absentes ou illisibles. Vérifiez les fichiers du dossier reports.")
    elif page == "Méthode":
        methodology()
    with st.container(key="footer"):
        st.caption(":material/info: YIL AI · Aide au classement documentaire · Validation humaine requise · Aucune décision de navigabilité")
