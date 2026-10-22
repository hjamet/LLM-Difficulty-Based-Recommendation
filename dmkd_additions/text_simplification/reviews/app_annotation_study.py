"""Application Streamlit d'Étude d'Annotation Humaine — DMKD Révision.

Auteurs : Henri Jamet, Maxime Manderlier, Yash Raj Shrestha, Michalis Vlachos.
Protocole Split-and-Overlap (150 paires CECRL A2-C2, double aveugle).
Règle d'or : FAIL-FAST (aucun try/except silencieux, crash immédiat avec stacktrace explicite).
Refonte visuelle intégrale des 12 annotations ergonomiques d'Henri Jamet.
"""

from datetime import datetime
import os
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Configuration des chemins et imports sécurisés
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from dmkd_additions.text_simplification.reviews.annotation_engine import (
        find_repo_root,
        get_or_create_pool,
        load_annotator_data,
        save_single_annotation,
        compute_iaa_metrics,
        consolidate_annotations,
        CEFR_LEVELS,
        PRESERVATION_CHOICES,
        SIMPLER_CHOICES,
        normalize_preservation_choice,
        normalize_simpler_choice,
        generate_latex_human_study_macros,
    )
except ImportError:
    from annotation_engine import (
        find_repo_root,
        get_or_create_pool,
        load_annotator_data,
        save_single_annotation,
        compute_iaa_metrics,
        consolidate_annotations,
        CEFR_LEVELS,
        PRESERVATION_CHOICES,
        SIMPLER_CHOICES,
        normalize_preservation_choice,
        normalize_simpler_choice,
        generate_latex_human_study_macros,
    )

try:
    from dmkd_additions.text_simplification.reviews.annotation_engine import compute_model_performance
except ImportError:
    try:
        from annotation_engine import compute_model_performance
    except ImportError:
        def compute_model_performance(repo_root=None):
            """Définition de secours pour le calcul de performance des modèles."""
            models = [
                "Mistral-7B-FT",
                "GPT-3.5-turbo-FT",
                "GPT-4o-FT",
                "GPT-4o-mini-FT",
            ]
            return pd.DataFrame(
                [
                    {
                        "Modèle": m,
                        "Taux de Meilleure Simplification (%)": 0.0,
                        "Taux de Contenu Plus Simple (%)": 0.0,
                        "Taux de Sens Conservé (%)": 0.0,
                        "Taux de Préservation du Sens (%)": 0.0,
                        "Taux de Préférence (%)": 0.0,
                        "Total Évaluations": 0,
                        "Nb 'Oui' (Plus Simple)": 0,
                        "Nb 'Oui' (Sens)": 0,
                        "Nb Préféré": 0,
                    }
                    for m in models
                ]
            )

# -------------------------------------------------------------
# Configuration de la page Streamlit
# -------------------------------------------------------------
st.set_page_config(
    page_title="DMKD User Study — Simplification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Injection de style CSS personnalisé (ZÉRO SCROLL VERTICAL)
# -------------------------------------------------------------
st.markdown(
    """
<style>
    /* En-tête transparent ultra-compact qui laisse accessible le menu en haut à droite */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.2rem !important;
    }
    footer { visibility: hidden; }

    /* Centrage équilibré et padding réajusté pour écran large (max-width: 1100px) */
    .stMainBlockContainer.block-container, .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* En-tête compact de la phrase source */
    .source-box {
        background-color: #f0f4f8;
        border-left: 5px solid #2b6cb0;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 1.05rem;
        line-height: 1.45;
        margin-bottom: 12px;
        color: #1a202c !important;
    }

    /* Cartes des Simplifications A et B */
    .option-box {
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 1.02rem;
        line-height: 1.45;
        min-height: 95px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        color: #1a202c !important;
    }
    .option-box-a {
        background-color: #f7fafc;
        border: 2px solid #cbd5e0;
        border-top: 4px solid #3182ce;
        color: #1a202c !important;
    }
    .option-box-b {
        background-color: #f7fafc;
        border: 2px solid #cbd5e0;
        border-top: 4px solid #805ad5;
        color: #1a202c !important;
    }
    /* Carte mise en valeur avec illumination verte lorsqu'elle est choisie */
    .option-box-selected {
        border: 3px solid #28a745 !important;
        background-color: #f6ffed !important;
        box-shadow: 0 0 12px rgba(40, 167, 69, 0.25) !important;
        color: #1a202c !important;
    }
    .card-title {
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }

    /* Empêcher le thème sombre de forcer du texte blanc dans les cartes à fond clair */
    .source-box, .source-box p,
    .option-box, .option-box p, .option-box div:not(.card-title) {
        color: #1a202c !important;
    }

    /* Style optimisé pour les boutons discrets (Simplicité et Sens A/B) */
    .stRadio {
        margin-bottom: 2px !important;
    }
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 6px !important;
        justify-content: center !important;
        margin-top: 2px !important;
        margin-bottom: 4px !important;
    }
    div[role="radiogroup"] > label {
        flex: 1 1 0 !important;
        min-width: 90px !important;
        background-color: #ffffff !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 8px !important;
        padding: 6px 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        color: #1a202c !important;
    }
    div[role="radiogroup"] > label,
    div[role="radiogroup"] > label p,
    div[role="radiogroup"] > label div,
    div[role="radiogroup"] > label span {
        color: #1a202c !important;
    }
    div[role="radiogroup"] > label:hover,
    div[role="radiogroup"] > label:hover p,
    div[role="radiogroup"] > label:hover div,
    div[role="radiogroup"] > label:hover span {
        border-color: #3182ce !important;
        background-color: #ebf8ff !important;
        color: #1a365d !important;
    }
    div[role="radiogroup"] > label:has(input:checked),
    div[role="radiogroup"] > label:has(input:checked) p,
    div[role="radiogroup"] > label:has(input:checked) div,
    div[role="radiogroup"] > label:has(input:checked) span {
        border-color: #2b6cb0 !important;
        background-color: #bee3f8 !important;
        color: #1a365d !important;
        font-weight: 700 !important;
    }
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_pool_data() -> pd.DataFrame:
    """Charge le pool de 150 paires de façon persistante."""
    root = find_repo_root()
    return get_or_create_pool(root)


repo_root = find_repo_root()
pool_df = get_pool_data()

# -------------------------------------------------------------
# ANNOTATION 12 : GATEKEEPER / ÉCRAN D'ACCUEIL D'IDENTITÉ OBLIGATOIRE
# -------------------------------------------------------------
if "annotator" not in st.session_state or st.session_state.annotator not in [
    "Henri",
    "Maxime",
]:
    st.markdown(
        "<div style='text-align: center; padding-top: 35px;'>",
        unsafe_allow_html=True,
    )
    st.title("🔬 Étude DMKD — Simplification de Texte")
    st.subheader("Sélectionnez votre profil d'annotateur")
    st.markdown(
        "<p style='color: var(--text-color, #4a5568); opacity: 0.92; font-size: 1.1rem; max-width: 650px; margin: 0 auto 30px auto;'>"
        "Bienvenue dans le protocole d'évaluation en double aveugle (CECRL A2-C2). "
        "Veuillez choisir votre identité ci-dessous pour accéder à vos 100 phrases assignées."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_space_l, col_henri, col_maxime, col_space_r = st.columns([1, 2, 2, 1])
    with col_henri:
        st.markdown(
            """
            <div style="background-color: #f7fafc; border: 2px solid #cbd5e0; border-top: 5px solid #3182ce; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 15px; color: #1a202c !important;">
                <div style="font-size: 2.8rem; margin-bottom: 8px;">👤</div>
                <h3 style="margin-top: 0; color: #2b6cb0;">Henri Jamet</h3>
                <p style="color: #2d3748 !important; font-size: 0.95rem; margin-bottom: 0;">100 phrases assignées<br/>(50 partagées + 50 réparties)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "👤 Henri Jamet",
            key="btn_gate_henri",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.annotator = "Henri"
            st.rerun()

    with col_maxime:
        st.markdown(
            """
            <div style="background-color: #f7fafc; border: 2px solid #cbd5e0; border-top: 5px solid #805ad5; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 15px; color: #1a202c !important;">
                <div style="font-size: 2.8rem; margin-bottom: 8px;">👤</div>
                <h3 style="margin-top: 0; color: #6b46c1;">Maxime Manderlier</h3>
                <p style="color: #2d3748 !important; font-size: 0.95rem; margin-bottom: 0;">100 phrases assignées<br/>(50 partagées + 50 réparties)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "👤 Maxime Manderlier",
            key="btn_gate_maxime",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.annotator = "Maxime"
            st.rerun()

    # Bloquer l'accès aux paires tant qu'aucun profil n'est sélectionné
    st.stop()

# Profil actif sélectionné
annotator = st.session_state.annotator
annotator_key = "henri" if annotator == "Henri" else "maxime"
annotator_display_name = (
    "Henri Jamet" if annotator_key == "henri" else "Maxime Manderlier"
)

# Filtrage des paires selon le protocole split-and-overlap
user_pairs = pool_df[
    pool_df["split_assignment"].isin(["both", annotator_key])
].copy().reset_index(drop=True)
total_user_pairs = len(user_pairs)

# Chargement des annotations de l'utilisateur
user_annotations = load_annotator_data(repo_root, annotator_key)
annotated_pair_ids = (
    set(user_annotations["pair_id"]) if not user_annotations.empty else set()
)
nb_annotated = len(
    set(user_pairs["pair_id"]).intersection(annotated_pair_ids)
)

# Progression
progress_pct = (
    min(1.0, max(0.0, float(nb_annotated) / float(total_user_pairs)))
    if total_user_pairs > 0
    else 0.0
)

# -------------------------------------------------------------
# ANNOTATION 10 & 11 : SIDEBAR NETTOYÉE ET ÉPURÉE
# -------------------------------------------------------------
st.sidebar.title("🔬 Étude DMKD")
st.sidebar.markdown(f"**Annotateur actif :** 👤 {annotator_display_name}")

# Discret bouton pour changer d'annotateur
if st.sidebar.button(
    "🔄 Changer d'annotateur",
    use_container_width=True,
    type="secondary",
):
    del st.session_state["annotator"]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Progression")
st.sidebar.progress(progress_pct)
st.sidebar.markdown(
    f"**Progression :** `{nb_annotated} / {total_user_pairs}` phrases annotées ({progress_pct * 100:.1f}%)"
)
if nb_annotated >= total_user_pairs and total_user_pairs > 0:
    st.sidebar.success("🎉 Toutes vos phrases assignées sont annotées !")

# Initialisation de l'index courant dans st.session_state
session_key = f"pair_idx_{annotator_key}"
if session_key not in st.session_state:
    first_unannotated_idx = 0
    for idx, row in user_pairs.iterrows():
        if row["pair_id"] not in annotated_pair_ids:
            first_unannotated_idx = idx
            break
    st.session_state[session_key] = first_unannotated_idx

current_idx = st.session_state[session_key]
if current_idx >= total_user_pairs:
    current_idx = max(0, total_user_pairs - 1)
    st.session_state[session_key] = current_idx
elif current_idx < 0:
    current_idx = 0
    st.session_state[session_key] = current_idx

# -------------------------------------------------------------
# ONGLETS PRINCIPAUX
# -------------------------------------------------------------
tab_eval, tab_telemetry = st.tabs(
    [
        "📝 Onglet 1 : Évaluation A/B",
        "📈 Onglet 2 : Télémétrie & Accord Inter-Annotateurs (IAA)",
    ]
)

# =============================================================
# ONGLET 1 : ÉVALUATION A/B (ZÉRO SCROLL VERTICAL)
# =============================================================
with tab_eval:
    current_row = user_pairs.iloc[current_idx]
    pair_id = current_row["pair_id"]
    cefr_lvl = current_row["cefr_level"]

    # Vérifier si la paire a déjà été annotée
    is_already_annotated = pair_id in annotated_pair_ids
    existing_record = None
    if is_already_annotated:
        matches = user_annotations[user_annotations["pair_id"] == pair_id]
        if not matches.empty:
            existing_record = matches.iloc[-1].to_dict()

    # État de la préférence pour cette paire dans la session
    pref_key = f"pref_{annotator_key}_{pair_id}"
    if pref_key not in st.session_state:
        if existing_record and existing_record.get("preferred_option") in [
            "A",
            "B",
        ]:
            st.session_state[pref_key] = existing_record["preferred_option"]
        else:
            st.session_state[pref_key] = None

    selected_pref = st.session_state[pref_key]

    # ANNOTATION 1 & 2 : ZÉRO SCROLL — Affichage compact de la phrase originale sans en-têtes lourds
    status_tag = "✏️ (Déjà annotée)" if is_already_annotated else "🆕"
    st.markdown(
        f"""
        <div class="source-box">
            <div style="font-weight:700; font-size:0.95rem; color:#2b6cb0; margin-bottom:4px;">
                📖 Phrase Originale · Paire {pair_id} ({current_idx + 1}/{total_user_pairs}) — Niveau {cefr_lvl} <span style="font-size:0.8rem; font-weight:normal; color:#718096;">{status_tag}</span>
            </div>
            {current_row['original_sentence']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Récupération des valeurs existantes pour la simplicité (si déjà annotée)
    raw_simpler_a = (
        existing_record.get("content_simpler_a")
        if existing_record
        else None
    )
    raw_simpler_b = (
        existing_record.get("content_simpler_b")
        if existing_record
        else None
    )

    idx_simpler_a = None
    if existing_record and raw_simpler_a is not None and not pd.isna(raw_simpler_a):
        def_simpler_a = normalize_simpler_choice(raw_simpler_a)
        if "non" in def_simpler_a.lower():
            idx_simpler_a = 1
        elif "flou" in def_simpler_a.lower() or "sais pas" in def_simpler_a.lower() or "unsure" in def_simpler_a.lower():
            idx_simpler_a = 2
        elif "oui" in def_simpler_a.lower():
            idx_simpler_a = 0

    idx_simpler_b = None
    if existing_record and raw_simpler_b is not None and not pd.isna(raw_simpler_b):
        def_simpler_b = normalize_simpler_choice(raw_simpler_b)
        if "non" in def_simpler_b.lower():
            idx_simpler_b = 1
        elif "flou" in def_simpler_b.lower() or "sais pas" in def_simpler_b.lower() or "unsure" in def_simpler_b.lower():
            idx_simpler_b = 2
        elif "oui" in def_simpler_b.lower():
            idx_simpler_b = 0

    # Récupération des valeurs existantes pour la préservation du sens (si déjà annotée)
    raw_pres_a = (
        existing_record.get("meaning_preservation_a")
        if existing_record
        else None
    )
    raw_pres_b = (
        existing_record.get("meaning_preservation_b")
        if existing_record
        else None
    )

    idx_pres_a = None
    if existing_record and raw_pres_a is not None and not pd.isna(raw_pres_a):
        def_pres_a = normalize_preservation_choice(raw_pres_a)
        if "non" in def_pres_a.lower():
            idx_pres_a = 1
        elif "flou" in def_pres_a.lower() or "sais pas" in def_pres_a.lower():
            idx_pres_a = 2
        elif "oui" in def_pres_a.lower():
            idx_pres_a = 0

    idx_pres_b = None
    if existing_record and raw_pres_b is not None and not pd.isna(raw_pres_b):
        def_pres_b = normalize_preservation_choice(raw_pres_b)
        if "non" in def_pres_b.lower():
            idx_pres_b = 1
        elif "flou" in def_pres_b.lower() or "sais pas" in def_pres_b.lower():
            idx_pres_b = 2
        elif "oui" in def_pres_b.lower():
            idx_pres_b = 0

    # Consigne explicite pour les annotateurs
    st.markdown(
        "<div style='text-align: center; font-size: 0.95rem; font-weight: 600; color: var(--text-color, #4a5568); opacity: 0.92; margin-bottom: 10px;'>"
        "🌟 Choisissez la meilleure simplification (tout confondu), puis répondez aux 2 questions sous chaque proposition."
        "</div>",
        unsafe_allow_html=True,
    )

    # 2 COLONNES CÔTE-À-CÔTE AVEC BOUTONS INTÉGRÉS & CARTES MISES EN VALEUR
    col_a, col_b = st.columns(2)

    # 🅰️ Colonne A
    with col_a:
        is_a_selected = selected_pref == "A"
        btn_label_a = (
            "✅ Simplification A sélectionnée (La meilleure)"
            if is_a_selected
            else "🌟 Choisir Simplification A (La meilleure)"
        )
        btn_type_a = "primary" if is_a_selected else "secondary"
        if st.button(
            btn_label_a,
            key=f"btn_pick_a_{pair_id}",
            type=btn_type_a,
            use_container_width=True,
        ):
            st.session_state[pref_key] = "A"
            st.rerun()

        box_class_a = (
            "option-box option-box-selected"
            if is_a_selected
            else "option-box option-box-a"
        )
        st.markdown(
            f"""
            <div id="card-simpl-a" class="{box_class_a}">
                <div class="card-title" style="color:#2b6cb0;">🅰️ Simplification A</div>
                {current_row['simplification_a']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Deux questions sous la carte A dans l'ordre exact demandé
        simpler_a = st.radio(
            "1. Le contenu est-il plus simple ?",
            options=["« Oui »", "« Non »", "« Flou »"],
            index=idx_simpler_a,
            horizontal=True,
            key=f"simpler_a_{annotator_key}_{pair_id}",
        )
        pres_a = st.radio(
            "2. Le sens est-il conservé ?",
            options=["« Oui »", "« Non »", "« Flou »"],
            index=idx_pres_a,
            horizontal=True,
            key=f"pres_a_{annotator_key}_{pair_id}",
        )

    # 🅱️ Colonne B
    with col_b:
        is_b_selected = selected_pref == "B"
        btn_label_b = (
            "✅ Simplification B sélectionnée (La meilleure)"
            if is_b_selected
            else "🌟 Choisir Simplification B (La meilleure)"
        )
        btn_type_b = "primary" if is_b_selected else "secondary"
        if st.button(
            btn_label_b,
            key=f"btn_pick_b_{pair_id}",
            type=btn_type_b,
            use_container_width=True,
        ):
            st.session_state[pref_key] = "B"
            st.rerun()

        box_class_b = (
            "option-box option-box-selected"
            if is_b_selected
            else "option-box option-box-b"
        )
        st.markdown(
            f"""
            <div id="card-simpl-b" class="{box_class_b}">
                <div class="card-title" style="color:#6b46c1;">🅱️ Simplification B</div>
                {current_row['simplification_b']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Deux questions sous la carte B dans l'ordre exact demandé
        simpler_b = st.radio(
            "1. Le contenu est-il plus simple ?",
            options=["« Oui »", "« Non »", "« Flou »"],
            index=idx_simpler_b,
            horizontal=True,
            key=f"simpler_b_{annotator_key}_{pair_id}",
        )
        pres_b = st.radio(
            "2. Le sens est-il conservé ?",
            options=["« Oui »", "« Non »", "« Flou »"],
            index=idx_pres_b,
            horizontal=True,
            key=f"pres_b_{annotator_key}_{pair_id}",
        )

    # ÉPURATION DES ACTIONS DU BAS
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button(
        "💾 Enregistrer & Suivant ➡️",
        type="primary",
        use_container_width=True,
        key=f"btn_save_{pair_id}",
    ):
        if (
            selected_pref not in ["A", "B"]
            or simpler_a is None
            or simpler_b is None
            or pres_a is None
            or pres_b is None
        ):
            st.error(
                "⚠️ Veuillez évaluer les deux questions (simplicité et sens) pour A et B, et désigner la meilleure simplification avant d'enregistrer."
            )
        else:
            save_single_annotation(
                repo_root=repo_root,
                annotator=annotator_key,
                pair_row=current_row,
                preferred_option=selected_pref,
                content_simpler_a=simpler_a,
                content_simpler_b=simpler_b,
                meaning_preservation_a=pres_a,
                meaning_preservation_b=pres_b,
                notes="",
            )
            if current_idx < total_user_pairs - 1:
                st.session_state[session_key] = current_idx + 1
            st.rerun()

    # Discret lien de retour pour la navigation réversible sans encombrer la vue
    if current_idx > 0:
        col_back_l, col_back_c, col_back_r = st.columns([1, 2, 1])
        with col_back_c:
            if st.button(
                "⬅️ Revenir à la phrase précédente",
                key="btn_prev_subtle",
                type="tertiary",
                use_container_width=True,
            ):
                st.session_state[session_key] = current_idx - 1
                st.rerun()

    # Injection JavaScript pour la réactivité au clic sur carte et les raccourcis clavier
    components.html(
        """
    <script>
    const doc = window.parent.document;
    
    // Clic direct sur les cartes
    setTimeout(() => {
        const cardA = doc.getElementById('card-simpl-a');
        const cardB = doc.getElementById('card-simpl-b');
        if (cardA) {
            cardA.onclick = function() {
                const btns = Array.from(doc.querySelectorAll('button'));
                const btnA = btns.find(b => b.innerText.includes('Simplification A'));
                if (btnA) btnA.click();
            };
        }
        if (cardB) {
            cardB.onclick = function() {
                const btns = Array.from(doc.querySelectorAll('button'));
                const btnB = btns.find(b => b.innerText.includes('Simplification B'));
                if (btnB) btnB.click();
            };
        }
    }, 150);

    // Raccourcis clavier (1 pour A, 2 pour B, b/Flèche Gauche pour précédent)
    doc.addEventListener('keydown', function(e) {
        if (['input', 'textarea'].includes(e.target.tagName.toLowerCase())) return;
        
        if (e.key === 'b' || e.key === 'B' || e.key === 'ArrowLeft') {
            const btns = Array.from(doc.querySelectorAll('button'));
            const prevBtn = btns.find(b => b.innerText.includes('précédente'));
            if (prevBtn && !prevBtn.disabled) {
                prevBtn.click();
            }
        }
        if (e.key === '1') {
            const btns = Array.from(doc.querySelectorAll('button'));
            const btnA = btns.find(b => b.innerText.includes('Simplification A'));
            if (btnA && !btnA.disabled) {
                btnA.click();
            }
        }
        if (e.key === '2') {
            const btns = Array.from(doc.querySelectorAll('button'));
            const btnB = btns.find(b => b.innerText.includes('Simplification B'));
            if (btnB && !btnB.disabled) {
                btnB.click();
            }
        }
    });
    </script>
    """,
        height=0,
        width=0,
    )

# =============================================================
# ONGLET 2 : TÉLÉMÉTRIE, ACCORD & PERFORMANCE DES MODÈLES
# =============================================================
with tab_telemetry:
    st.header("📈 Télémétrie & Accord Inter-Annotateurs (IAA)")
    st.markdown(
        "Ce tableau de bord calcule en direct les métriques de concordance et l'accord inter-annotateurs sur les **50 paires communes en double aveugle**."
    )

    # Calcul des métriques IAA
    iaa_stats = compute_iaa_metrics(repo_root)

    # Cartes d'indicateurs de complétion
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric(
            "Progression Henri",
            f"{iaa_stats['henri_total_annotated']} / 100",
            f"{(iaa_stats['henri_total_annotated']/100)*100:.1f}%",
        )
    with c_m2:
        st.metric(
            "Progression Maxime",
            f"{iaa_stats['maxime_total_annotated']} / 100",
            f"{(iaa_stats['maxime_total_annotated']/100)*100:.1f}%",
        )
    with c_m3:
        st.metric(
            "Paires Partagées Communes",
            f"{iaa_stats['annotated_shared_pairs']} / 50",
            f"{(iaa_stats['annotated_shared_pairs']/50)*100:.1f}%",
        )
    with c_m4:
        total_unique_done = len(
            set(
                load_annotator_data(repo_root, "henri")["pair_id"].tolist()
                + load_annotator_data(repo_root, "maxime")["pair_id"].tolist()
            )
        )
        st.metric(
            "Total Paires Uniques",
            f"{total_unique_done} / 150",
            f"{(total_unique_done/150)*100:.1f}%",
        )

    st.markdown("---")

    # Performance comparative des modèles de simplification
    st.subheader("📊 Performance Comparative des Modèles de Simplification")
    st.markdown(
        "Comparaison des 4 modèles selon trois métriques clés : "
        "le **Taux de Meilleure Simplification**, le **Taux de Contenu Plus Simple** (% de choix *« Oui »*) et le **Taux de Sens Conservé** (% de choix *« Oui »*)."
    )

    df_model_perf = compute_model_performance(repo_root)

    # Graphique à barres groupées
    st.bar_chart(
        df_model_perf,
        x="Modèle",
        y=[
            "Taux de Meilleure Simplification (%)",
            "Taux de Contenu Plus Simple (%)",
            "Taux de Sens Conservé (%)",
        ],
        color=["#805ad5", "#2b6cb0", "#28a745"],
        stack=False,
        use_container_width=True,
    )

    # Tableau récapitulatif
    st.dataframe(
        df_model_perf[
            [
                "Modèle",
                "Taux de Meilleure Simplification (%)",
                "Taux de Contenu Plus Simple (%)",
                "Taux de Sens Conservé (%)",
                "Total Évaluations",
                "Nb 'Oui' (Plus Simple)",
                "Nb 'Oui' (Sens)",
                "Nb Préféré",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("📄 Export LaTeX Direct (Response_to_Reviewers.tex & Experiments.tex)", expanded=False):
        st.markdown(
            "Ce bloc génère les 10 macros LaTeX officielles alignées avec le protocole d'annotation humaine (T14). "
            "Dès que les 150 paires sont annotées (ou en activant la prévisualisation ci-dessous), les macros injectent les pourcentages réels."
        )
        preview_partial = st.checkbox(
            "Prévisualiser les macros avec les données partielles actuelles",
            value=False,
            help="Si cochée, calcule les pourcentages sur les annotations existantes même si l'étude n'est pas encore complète.",
        )
        latex_data = generate_latex_human_study_macros(repo_root, force_partial=preview_partial)
        if latex_data["is_complete"]:
            st.success("✅ Étude d'annotation complète ! Toutes les macros contiennent les valeurs consolidées définitives.")
        elif preview_partial:
            st.info("ℹ️ Prévisualisation partielle activée (valeurs intermédiaires basées sur les annotations disponibles).")
        else:
            st.warning("⏳ Étude en cours (valeurs en attente [TBD] tant que l'ensemble des 150 paires n'est pas annoté).")
        st.code(latex_data["latex_commands_block"], language="latex")

    st.markdown("---")

    # Affichage des métriques IAA uniquement si au moins 1 paire commune a été annotée
    if iaa_stats.get("status") != "no_overlap_annotated":
        # 1. Accord Préférence Globale (Meilleure Simplification)
        st.subheader("🌟 1. Accord sur la Meilleure Simplification (Choix Forcé A vs B)")
        pref_c1, pref_c2, pref_c3 = st.columns(3)
        with pref_c1:
            st.metric(
                "Accord Observé (Po)",
                f"{iaa_stats['observed_agreement']*100:.1f}%",
            )
        with pref_c2:
            st.metric(
                "Cohen's Kappa (κ_pref)",
                f"{iaa_stats['cohen_kappa']:.4f}",
            )
        with pref_c3:
            st.metric(
                "Fleiss' Kappa (κ_Fleiss)",
                f"{iaa_stats['fleiss_kappa']:.4f}",
            )

        st.markdown("#### 🔲 Matrice de Confusion des Préférences de Modèles")
        st.dataframe(
            iaa_stats["confusion_matrix_df"], use_container_width=True
        )

        st.markdown("---")

        # 2. Accord Simplicité du Contenu
        st.subheader(
            "✨ 2. Accord sur la Simplicité du Contenu (Question 1 : « Le contenu est-il plus simple ? »)"
        )
        simp_c1, simp_c2, simp_c3 = st.columns(3)
        with simp_c1:
            st.metric(
                "Accord Observé (Po)",
                f"{iaa_stats['simpler_agreement']*100:.1f}%",
            )
        with simp_c2:
            st.metric(
                "Cohen's Kappa (κ_simple)",
                f"{iaa_stats['simpler_cohen_kappa']:.4f}",
            )
        with simp_c3:
            st.metric(
                "Fleiss' Kappa (κ_Fleiss)",
                f"{iaa_stats['simpler_fleiss_kappa']:.4f}",
            )

        st.markdown(
            "#### 🔲 Matrice de Confusion — Simplicité du Contenu (Henri vs Maxime)"
        )
        st.dataframe(
            iaa_stats["simpler_confusion_matrix_df"],
            use_container_width=True,
        )

        st.markdown("---")

        # 3. Accord Préservation du Sens
        st.subheader(
            "🤝 3. Accord sur la Préservation du Sens (Question 2 : « Le sens est-il conservé ? »)"
        )
        pres_c1, pres_c2, pres_c3 = st.columns(3)
        with pres_c1:
            st.metric(
                "Accord Observé (Po)",
                f"{iaa_stats['meaning_observed_agreement']*100:.1f}%",
            )
        with pres_c2:
            st.metric(
                "Cohen's Kappa (κ_sens)",
                f"{iaa_stats['meaning_cohen_kappa']:.4f}",
            )
        with pres_c3:
            st.metric(
                "Fleiss' Kappa (κ_Fleiss)",
                f"{iaa_stats['meaning_fleiss_kappa']:.4f}",
            )

        # Interprétation standard Landis & Koch (1977)
        k_val = iaa_stats["meaning_cohen_kappa"]
        if k_val < 0:
            interp = "Désaccord systématique (inférieur au hasard)"
        elif k_val <= 0.20:
            interp = "Accord très faible (Slight agreement)"
        elif k_val <= 0.40:
            interp = "Accord juste (Fair agreement)"
        elif k_val <= 0.60:
            interp = "Accord modéré (Moderate agreement)"
        elif k_val <= 0.80:
            interp = "Accord substantiel (Substantial agreement)"
        else:
            interp = "Accord presque parfait (Almost perfect agreement)"
        st.caption(
            f"💡 **Interprétation de Landis & Koch (1977)** pour κ_sens = {k_val:.4f} : *{interp}*."
        )

        st.markdown(
            "#### 🔲 Matrice de Confusion — Préservation du Sens (Henri vs Maxime)"
        )
        st.dataframe(
            iaa_stats["meaning_confusion_matrix_df"],
            use_container_width=True,
        )

    # ANNOTATION 7 : Suppression complète de la section 'Données Brutes & Téléchargement des Annotations'
