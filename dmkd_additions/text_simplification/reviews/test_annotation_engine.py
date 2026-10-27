import os
from pathlib import Path
import sys

# Assurer la présence du repo_root dans sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd

from dmkd_additions.text_simplification.reviews.annotation_engine import (
    find_repo_root,
    load_raw_datasets,
    generate_annotation_pool,
    get_or_create_pool,
    detect_pool_anomalies,
    save_atomic_dataframe,
    save_single_annotation,
    load_annotator_data,
    compute_fleiss_kappa,
    compute_krippendorff_alpha_ordinal,
    compute_iaa_metrics,
    consolidate_annotations,
    compute_model_performance,
    generate_latex_human_study_macros,
    CEFR_LEVELS,
    PRESERVATION_CHOICES,
    SIMPLER_CHOICES,
    normalize_preservation_choice,
    normalize_simpler_choice,
    get_next_unannotated_index,
)


def test_detect_pool_anomalies():
    """Vérifie la détection d'anomalies sur cas synthétiques et l'absence totale d'anomalie sur le pool réel."""
    repo_root = find_repo_root()
    pool = get_or_create_pool(repo_root)
    assert len(pool) == 150

    pool_anom = detect_pool_anomalies(pool)
    assert "is_nan_pair" in pool_anom.columns
    assert "is_prompt_leak" in pool_anom.columns
    assert "is_identical_pair" in pool_anom.columns
    assert "is_clean_pair" in pool_anom.columns

    # 1. Le pool d'annotation réel ne doit contenir AUCUNE anomalie (0 nan, 0 leak, 0 identical)
    assert pool_anom["is_nan_pair"].sum() == 0, f"NaN inattendus : {pool_anom[pool_anom['is_nan_pair']]['pair_id'].tolist()}"
    assert pool_anom["is_prompt_leak"].sum() == 0, f"Prompt leak inattendu : {pool_anom[pool_anom['is_prompt_leak']]['pair_id'].tolist()}"
    assert pool_anom["is_identical_pair"].sum() == 0, f"Identiques inattendus : {pool_anom[pool_anom['is_identical_pair']]['pair_id'].tolist()}"
    assert pool_anom["is_clean_pair"].sum() == 150, f"Total clean = {pool_anom['is_clean_pair'].sum()} (attendu: 150)"

    # 2. Vérification sur un échantillon synthétique avec anomalies
    synthetic = pd.DataFrame([
        {"pair_id": "T1", "simplification_a": "nan", "simplification_b": "Texte propre"},
        {"pair_id": "T2", "simplification_a": "[/INST] Leak", "simplification_b": "Texte propre"},
        {"pair_id": "T3", "simplification_a": "Texte identique", "simplification_b": "Texte identique"},
        {"pair_id": "T4", "simplification_a": "Texte A propre", "simplification_b": "Texte B propre"},
    ])
    synth_anom = detect_pool_anomalies(synthetic)
    assert synth_anom.loc[synth_anom["pair_id"] == "T1", "is_nan_pair"].iloc[0]
    assert synth_anom.loc[synth_anom["pair_id"] == "T2", "is_prompt_leak"].iloc[0]
    assert synth_anom.loc[synth_anom["pair_id"] == "T3", "is_identical_pair"].iloc[0]
    assert synth_anom.loc[synth_anom["pair_id"] == "T4", "is_clean_pair"].iloc[0]


def test_clean_vs_raw_metrics_on_henri_100():
    """Vérifie les métriques sur les 100 annotations réelles d'Henri et sa progression complète (100/100)."""
    repo_root = find_repo_root()
    reviews_dir = repo_root / "dmkd_additions" / "text_simplification" / "reviews"
    f_h = reviews_dir / "annotations_henri.csv"
    assert f_h.is_file(), "annotations_henri.csv doit être présent"

    df_henri = pd.read_csv(f_h)
    assert len(df_henri) == 100, f"annotations_henri.csv doit avoir 100 lignes, obtenu {len(df_henri)}"

    # Vérification de la progression (100/100 achevées, 0 restante)
    pool = pd.read_csv(reviews_dir / "annotation_pool_150.csv")
    user_pairs = pool[pool["split_assignment"].isin(["henri", "both"])].copy()
    assert len(user_pairs) == 100
    annotated_ids = set(df_henri["pair_id"])
    remaining = sorted(list(set(user_pairs["pair_id"]) - annotated_ids))
    assert remaining == [], f"Toutes les paires d'Henri doivent être annotées, restantes : {remaining}"

    # 1. Performance du modèle sur les 100 annotations
    perf = compute_model_performance(repo_root)
    assert len(perf) == 4
    perf_map = perf.set_index("Modèle").to_dict(orient="index")

    # Vérifications des chiffres connus sur le pool consolidé complet (200/200)
    assert perf_map["Mistral-7B-FT"]["preference_rate"] == 45.0
    assert perf_map["GPT-4o-FT"]["preference_rate"] == 58.0


def test_get_next_unannotated_index():
    """Vérifie la logique de navigation par saut vers la prochaine paire non annotée."""
    df_pairs = pd.DataFrame([
        {"pair_id": f"P{i:03d}"} for i in range(1, 6)
    ])  # P001, P002, P003, P004, P005

    # Cas 1 : Saut par-dessus les paires déjà remplies (actuel: 0=P001, P002 et P003 déjà annotées -> va à 3=P004)
    annotated_1 = {"P001", "P002", "P003"}
    assert get_next_unannotated_index(df_pairs, annotated_1, current_idx=0) == 3

    # Cas 2 : Bouclage au début (actuel: 3=P004, P005 annotée, mais P002 n'est pas annotée -> va à 1=P002)
    annotated_2 = {"P001", "P003", "P004", "P005"}
    assert get_next_unannotated_index(df_pairs, annotated_2, current_idx=3) == 1

    # Cas 3 : Cas 100% complet (toutes les paires sont annotées -> retourne min(current_idx + 1, len - 1))
    annotated_all = {"P001", "P002", "P003", "P004", "P005"}
    assert get_next_unannotated_index(df_pairs, annotated_all, current_idx=2) == 3
    assert get_next_unannotated_index(df_pairs, annotated_all, current_idx=4) == 4

    # Cas 4 : Cas bord avec DataFrame vide
    empty_df = pd.DataFrame(columns=["pair_id"])
    assert get_next_unannotated_index(empty_df, set(), 0) == 0


def test_engine_suite():
    """Suite complète de tests unitaires et d'intégration avec sauvegarde/restauration garantie."""
    repo_root = find_repo_root()
    assert (repo_root / "results").is_dir()
    assert (repo_root / "dmkd_additions").is_dir()

    reviews_dir = repo_root / "dmkd_additions" / "text_simplification" / "reviews"
    f_h = reviews_dir / "annotations_henri.csv"
    f_m = reviews_dir / "annotations_maxime.csv"
    f_c = reviews_dir / "annotations_consolidated.csv"

    # Sauvegarde octet par octet pour garantir un git status clean
    h_backup = f_h.read_bytes() if f_h.is_file() else None
    m_backup = f_m.read_bytes() if f_m.is_file() else None
    c_backup = f_c.read_bytes() if f_c.is_file() else None

    try:
        # Nettoyage temporaire pour tester les cas d'étude initiale
        if f_h.exists():
            f_h.unlink()
        if f_m.exists():
            f_m.unlink()
        if f_c.exists():
            f_c.unlink()

        # 1. Test load_raw_datasets
        df_hist, df_recent = load_raw_datasets(repo_root)
        assert len(df_hist) >= 150
        assert len(df_recent) == 500
        for lvl in CEFR_LEVELS:
            assert (df_hist["cefr_level"] == lvl).sum() >= 15
            assert (df_recent["cefr_level"] == lvl).sum() == 100

        # 2. Test generate_annotation_pool (50 both, 50 henri, 50 maxime)
        pool = generate_annotation_pool(repo_root, seed=42)
        assert len(pool) == 150
        assert (pool["split_assignment"] == "both").sum() == 50
        assert (pool["split_assignment"] == "henri").sum() == 50
        assert (pool["split_assignment"] == "maxime").sum() == 50
        assert (pool["competition"] == "historical").sum() == 75
        assert (pool["competition"] == "recent").sum() == 75
        for lvl in CEFR_LEVELS:
            assert (pool["cefr_level"] == lvl).sum() == 30
            lvl_sub = pool[pool["cefr_level"] == lvl]
            assert (lvl_sub["split_assignment"] == "both").sum() == 10
            assert (lvl_sub["split_assignment"] == "henri").sum() == 10
            assert (lvl_sub["split_assignment"] == "maxime").sum() == 10

        # 3. Test get_or_create_pool
        pool_loaded = get_or_create_pool(repo_root)
        pool_file = reviews_dir / "annotation_pool_150.csv"
        assert pool_file.is_file()
        assert len(pool_loaded) == 150

        # 4. Test Fleiss' Kappa
        mat = np.array([
            [3, 0],
            [0, 3],
            [2, 1],
            [1, 2],
            [3, 0],
        ])
        kappa = compute_fleiss_kappa(mat)
        assert -1.0 <= kappa <= 1.0

        mat_perfect = np.array([
            [2, 0],
            [0, 2],
            [2, 0],
        ])
        kappa_perf = compute_fleiss_kappa(mat_perfect)
        assert np.isclose(kappa_perf, 1.0)

        # 5. Test save_single_annotation (atomique) avec Simplicité et Sens
        test_annotator = "test_evaluator"
        test_file = reviews_dir / f"annotations_{test_annotator}.csv"
        if test_file.exists():
            test_file.unlink()

        row0 = pool_loaded.iloc[0]
        save_single_annotation(
            repo_root=repo_root,
            annotator=test_annotator,
            pair_row=row0,
            preferred_option="A",
            content_simpler_a=SIMPLER_CHOICES[0],
            content_simpler_b=SIMPLER_CHOICES[1],
            meaning_preservation_a=PRESERVATION_CHOICES[0],
            meaning_preservation_b=PRESERVATION_CHOICES[1],
            notes="Test note initial",
        )
        assert test_file.is_file()
        df_eval = load_annotator_data(repo_root, test_annotator)
        assert len(df_eval) == 1
        assert df_eval.iloc[0]["pair_id"] == row0["pair_id"]
        assert df_eval.iloc[0]["preferred_option"] == "A"
        assert df_eval.iloc[0]["content_simpler_a"] == SIMPLER_CHOICES[0]
        assert df_eval.iloc[0]["content_simpler_b"] == SIMPLER_CHOICES[1]
        assert df_eval.iloc[0]["meaning_preservation_a"] == PRESERVATION_CHOICES[0]
        assert df_eval.iloc[0]["meaning_preservation_b"] == PRESERVATION_CHOICES[1]

        # In-situ modification
        save_single_annotation(
            repo_root=repo_root,
            annotator=test_annotator,
            pair_row=row0,
            preferred_option="B",
            content_simpler_a=SIMPLER_CHOICES[2],
            content_simpler_b=SIMPLER_CHOICES[0],
            meaning_preservation_a=PRESERVATION_CHOICES[2],
            meaning_preservation_b=PRESERVATION_CHOICES[0],
            notes="Test note modifiee",
        )
        df_eval_mod = load_annotator_data(repo_root, test_annotator)
        assert len(df_eval_mod) == 1
        assert df_eval_mod.iloc[0]["preferred_option"] == "B"

        if test_file.exists():
            test_file.unlink()

        # 6. Test compute_iaa_metrics (empty)
        iaa_empty = compute_iaa_metrics(repo_root)
        assert iaa_empty["total_shared_pairs"] == 50

        # 7. Test compute_iaa_metrics with simulated shared annotations
        shared_rows = pool_loaded[pool_loaded["split_assignment"] == "both"].head(5)
        for _, srow in shared_rows.iterrows():
            save_single_annotation(
                repo_root=repo_root,
                annotator="henri",
                pair_row=srow,
                preferred_option="A",
                content_simpler_a=SIMPLER_CHOICES[0],
                content_simpler_b=SIMPLER_CHOICES[1],
                meaning_preservation_a=PRESERVATION_CHOICES[0],
                meaning_preservation_b=PRESERVATION_CHOICES[1],
            )
            m_choice = "A" if srow["pair_id"] != shared_rows.iloc[-1]["pair_id"] else "B"
            save_single_annotation(
                repo_root=repo_root,
                annotator="maxime",
                pair_row=srow,
                preferred_option=m_choice,
                content_simpler_a=SIMPLER_CHOICES[0],
                content_simpler_b=SIMPLER_CHOICES[1],
                meaning_preservation_a=PRESERVATION_CHOICES[0],
                meaning_preservation_b=PRESERVATION_CHOICES[1],
            )

        iaa_sim = compute_iaa_metrics(repo_root)
        assert iaa_sim["annotated_shared_pairs"] == 5
        assert iaa_sim["observed_agreement"] == 0.8
        assert iaa_sim["simpler_observed_agreement"] == 1.0
        assert iaa_sim["meaning_observed_agreement"] == 1.0

        # 8. Test compute_model_performance
        df_perf = compute_model_performance(repo_root)
        assert len(df_perf) == 4
        assert "Modèle" in df_perf.columns
        assert df_perf["Total Évaluations"].sum() > 0

        # 9. Test Krippendorff & macros LaTeX
        krip_perfect = compute_krippendorff_alpha_ordinal([0, 1, 2, 2, 1], [0, 1, 2, 2, 1])
        assert np.isclose(krip_perfect, 1.0)

        res_macros = generate_latex_human_study_macros(repo_root=repo_root, force_partial=True)
        assert "macros_dict" in res_macros
        assert len(res_macros["macros_dict"]) == 10

    finally:
        # Nettoyage et restauration stricte des fichiers réels
        if f_h.exists():
            f_h.unlink()
        if f_m.exists():
            f_m.unlink()
        if f_c.exists():
            f_c.unlink()

        if h_backup is not None:
            f_h.write_bytes(h_backup)
        if m_backup is not None:
            f_m.write_bytes(m_backup)
        if c_backup is not None:
            f_c.write_bytes(c_backup)


def run_tests():
    test_detect_pool_anomalies()
    test_clean_vs_raw_metrics_on_henri_100()
    test_get_next_unannotated_index()
    test_engine_suite()
    print("=== TOUS LES TESTS DE L'ENGINE SONT PASSÉS AVEC SUCCÈS ! ===")


if __name__ == "__main__":
    run_tests()
