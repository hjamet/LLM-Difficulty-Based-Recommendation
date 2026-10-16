import os
import sys
from pathlib import Path

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
    save_atomic_dataframe,
    save_single_annotation,
    load_annotator_data,
    compute_fleiss_kappa,
    compute_iaa_metrics,
    consolidate_annotations,
    compute_model_performance,
    CEFR_LEVELS,
    PRESERVATION_CHOICES,
)

def run_tests():
    print("=== Démarrage des tests unitaires de annotation_engine.py ===")
    
    # 1. Test find_repo_root
    repo_root = find_repo_root()
    print(f"1. find_repo_root: OK -> {repo_root}")
    assert (repo_root / "results").is_dir()
    assert (repo_root / "dmkd_additions").is_dir()
    
    # 2. Test load_raw_datasets
    df_hist, df_recent = load_raw_datasets(repo_root)
    print(f"2. load_raw_datasets: OK -> hist={len(df_hist)}, recent={len(df_recent)}")
    assert len(df_hist) >= 150
    assert len(df_recent) == 500
    for lvl in CEFR_LEVELS:
        assert (df_hist['cefr_level'] == lvl).sum() >= 15
        assert (df_recent['cefr_level'] == lvl).sum() == 100
        
    # 3. Test generate_annotation_pool (50 both, 50 henri, 50 maxime)
    pool = generate_annotation_pool(repo_root, seed=42)
    print(f"3. generate_annotation_pool: OK -> {len(pool)} paires")
    assert len(pool) == 150
    assert (pool['split_assignment'] == 'both').sum() == 50
    assert (pool['split_assignment'] == 'henri').sum() == 50
    assert (pool['split_assignment'] == 'maxime').sum() == 50
    assert (pool['competition'] == 'historical').sum() == 75
    assert (pool['competition'] == 'recent').sum() == 75
    for lvl in CEFR_LEVELS:
        assert (pool['cefr_level'] == lvl).sum() == 30
        lvl_sub = pool[pool['cefr_level'] == lvl]
        assert (lvl_sub['split_assignment'] == 'both').sum() == 10
        assert (lvl_sub['split_assignment'] == 'henri').sum() == 10
        assert (lvl_sub['split_assignment'] == 'maxime').sum() == 10
        
    # 4. Test get_or_create_pool
    pool_loaded = get_or_create_pool(repo_root)
    print(f"4. get_or_create_pool: OK -> {len(pool_loaded)} paires persistées")
    pool_file = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / "annotation_pool_150.csv"
    assert pool_file.is_file()
    assert len(pool_loaded) == 150
    
    # 5. Test Fleiss' Kappa on known manual case
    # Example from literature: 5 subjects, 3 raters, 2 categories
    mat = np.array([
        [3, 0],
        [0, 3],
        [2, 1],
        [1, 2],
        [3, 0]
    ])
    kappa = compute_fleiss_kappa(mat)
    print(f"5. compute_fleiss_kappa: OK -> kappa = {kappa:.4f}")
    assert -1.0 <= kappa <= 1.0
    
    # Perfect agreement test:
    mat_perfect = np.array([
        [2, 0],
        [0, 2],
        [2, 0]
    ])
    kappa_perf = compute_fleiss_kappa(mat_perfect)
    assert np.isclose(kappa_perf, 1.0)
    print("   Perfect agreement kappa test: OK (1.0000)")

    # 6. Test save_single_annotation (atomique) avec les 3 boutons discrets
    test_annotator = "test_evaluator"
    test_file = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / f"annotations_{test_annotator}.csv"
    if test_file.exists():
        test_file.unlink()
        
    row0 = pool_loaded.iloc[0]
    save_single_annotation(
        repo_root=repo_root,
        annotator=test_annotator,
        pair_row=row0,
        preferred_option="A",
        meaning_preservation_a=PRESERVATION_CHOICES[0],
        meaning_preservation_b=PRESERVATION_CHOICES[1],
        notes="Test note initial"
    )
    assert test_file.is_file()
    df_eval = load_annotator_data(repo_root, test_annotator)
    assert len(df_eval) == 1
    assert df_eval.iloc[0]["pair_id"] == row0["pair_id"]
    assert df_eval.iloc[0]["preferred_option"] == "A"
    assert df_eval.iloc[0]["meaning_preservation_a"] == PRESERVATION_CHOICES[0]
    assert df_eval.iloc[0]["meaning_preservation_b"] == PRESERVATION_CHOICES[1]
    
    # In-situ modification test (mode modification)
    save_single_annotation(
        repo_root=repo_root,
        annotator=test_annotator,
        pair_row=row0,
        preferred_option="B",
        meaning_preservation_a=PRESERVATION_CHOICES[2],
        meaning_preservation_b=PRESERVATION_CHOICES[0],
        notes="Test note modifiee"
    )
    df_eval_mod = load_annotator_data(repo_root, test_annotator)
    assert len(df_eval_mod) == 1, "La modification in-situ ne doit pas dupliquer la ligne !"
    assert df_eval_mod.iloc[0]["preferred_option"] == "B"
    assert df_eval_mod.iloc[0]["meaning_preservation_a"] == PRESERVATION_CHOICES[2]
    assert df_eval_mod.iloc[0]["meaning_preservation_b"] == PRESERVATION_CHOICES[0]
    print("6. save_single_annotation & in-situ update: OK")
    
    # Clean up test evaluator file
    if test_file.exists():
        test_file.unlink()

    # 7. Test compute_iaa_metrics
    # Test when no shared annotations exist
    iaa_empty = compute_iaa_metrics(repo_root)
    print(f"7. compute_iaa_metrics (empty): OK -> status={iaa_empty.get('status')}")
    assert iaa_empty["total_shared_pairs"] == 50

    # Test with simulated shared annotations between henri and maxime
    shared_rows = pool_loaded[pool_loaded['split_assignment'] == 'both'].head(5)
    for _, srow in shared_rows.iterrows():
        # Henri chooses option A
        save_single_annotation(
            repo_root=repo_root,
            annotator="henri",
            pair_row=srow,
            preferred_option="A",
            meaning_preservation_a=PRESERVATION_CHOICES[0],
            meaning_preservation_b=PRESERVATION_CHOICES[1],
        )
        # Maxime chooses option A on 4 out of 5
        m_choice = "A" if srow["pair_id"] != shared_rows.iloc[-1]["pair_id"] else "B"
        save_single_annotation(
            repo_root=repo_root,
            annotator="maxime",
            pair_row=srow,
            preferred_option=m_choice,
            meaning_preservation_a=PRESERVATION_CHOICES[0],
            meaning_preservation_b=PRESERVATION_CHOICES[1],
        )
        
    iaa_sim = compute_iaa_metrics(repo_root)
    print(f"8. compute_iaa_metrics (with 5 shared pairs): OK -> agreement={iaa_sim['observed_agreement']:.2f}, cohen_kappa_pref={iaa_sim['cohen_kappa']:.4f}, meaning_cohen_kappa={iaa_sim['meaning_cohen_kappa']:.4f}")
    assert iaa_sim["annotated_shared_pairs"] == 5
    assert iaa_sim["observed_agreement"] == 0.8
    assert iaa_sim["meaning_observed_agreement"] == 1.0
    assert "meaning_confusion_matrix_df" in iaa_sim
    assert "model_stats_df" in iaa_sim
    
    # Test compute_model_performance
    df_perf = compute_model_performance(repo_root)
    print(f"9. compute_model_performance: OK -> {len(df_perf)} modèles analysés")
    assert len(df_perf) == 4
    assert "Modèle" in df_perf.columns
    assert "Taux de Préservation du Sens (%)" in df_perf.columns
    assert "Taux de Préférence (%)" in df_perf.columns
    assert df_perf["Total Évaluations"].sum() > 0

    # Clean up simulated annotations
    f_h = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / "annotations_henri.csv"
    f_m = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / "annotations_maxime.csv"
    f_c = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / "annotations_consolidated.csv"
    if f_h.exists(): f_h.unlink()
    if f_m.exists(): f_m.unlink()
    if f_c.exists(): f_c.unlink()

    print("\n=== TOUS LES TESTS DE L'ENGINE SONT PASSÉS AVEC SUCCÈS ! ===")

if __name__ == "__main__":
    run_tests()
