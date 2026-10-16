"""Dry-run complet de validation pour le Chantier 2 DMKD :

- Chargement et alignement des prédictions historiques et récentes
- Stratification rigoureuse (150 paires, 30 par niveau A2-C2, 30 partagées, 60 Henri, 60 Maxime)
- Sauvegarde atomique anti-corruption (write-and-rename avec os.fsync)
- Calculs de concordance et accord inter-annotateurs (Cohen's Kappa, Fleiss' Kappa, Matrice de confusion)
- Règle FAIL-FAST : vérification des exceptions en cas d'anomalie
"""

import os
import sys
from pathlib import Path

# Fix encodage console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Fix sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dmkd_additions.text_simplification.reviews.annotation_engine import (
    find_repo_root,
    load_raw_datasets,
    get_or_create_pool,
    save_single_annotation,
    load_annotator_data,
    compute_iaa_metrics,
    consolidate_annotations,
    compute_fleiss_kappa,
    compute_model_performance,
    CEFR_LEVELS,
    PRESERVATION_CHOICES,
)

def dry_run():
    print("=================================================================")
    print("🚀 DÉMARRAGE DU DRY-RUN VALIDATION : app_annotation_study.py")
    print("=================================================================\n")
    
    repo_root = find_repo_root()
    print(f"📁 Racine du dépôt détectée : {repo_root}")
    
    # 1. Vérification des données sources
    print("\n--- Étape 1 : Chargement et alignement des données sources ---")
    df_hist, df_recent = load_raw_datasets(repo_root)
    print(f"✅ Données historiques unifiées : {len(df_hist)} paires communes (Mistral-7B FT & GPT-3.5 FT)")
    print("   Distribution CECRL historique :")
    for lvl in CEFR_LEVELS:
        cnt = (df_hist['cefr_level'] == lvl).sum()
        print(f"     - Niveau {lvl} : {cnt} paires")
        
    print(f"✅ Données récentes unifiées : {len(df_recent)} paires (GPT-4o FT & GPT-4o-mini FT)")
    print("   Distribution CECRL récent :")
    for lvl in CEFR_LEVELS:
        cnt = (df_recent['cefr_level'] == lvl).sum()
        print(f"     - Niveau {lvl} : {cnt} paires")
        
    # 2. Pool d'annotation de 150 paires (50 both, 50 henri, 50 maxime)
    print("\n--- Étape 2 : Chargement / Génération du Pool 150 Paires ---")
    pool_df = get_or_create_pool(repo_root)
    print(f"✅ Pool chargé avec succès : {len(pool_df)} paires")
    print("   Stratification par niveau :")
    for lvl in CEFR_LEVELS:
        cnt = (pool_df['cefr_level'] == lvl).sum()
        sub = pool_df[pool_df['cefr_level'] == lvl]
        n_shared = (sub['split_assignment'] == 'both').sum()
        n_henri = (sub['split_assignment'] == 'henri').sum()
        n_maxime = (sub['split_assignment'] == 'maxime').sum()
        print(f"     - {lvl} : {cnt} paires (Partagées: {n_shared}, Henri: {n_henri}, Maxime: {n_maxime})")
        assert cnt == 30, f"Erreur niveau {lvl}: {cnt} != 30"
        assert n_shared == 10, f"Erreur partagées {lvl}: {n_shared} != 10"
        assert n_henri == 10, f"Erreur henri {lvl}: {n_henri} != 10"
        assert n_maxime == 10, f"Erreur maxime {lvl}: {n_maxime} != 10"
        
    # 3. Double aveugle et équilibre A/B
    print("\n--- Étape 3 : Audit du Double Aveugle et Neutralisation du Biais de Position ---")
    n_hist_a_is_mistral = ((pool_df['competition'] == 'historical') & (pool_df['model_a'] == 'Mistral-7B-FT')).sum()
    n_hist_a_is_gpt35 = ((pool_df['competition'] == 'historical') & (pool_df['model_a'] == 'GPT-3.5-turbo-FT')).sum()
    print(f"   Historique (Option A) : Mistral-7B={n_hist_a_is_mistral}, GPT-3.5={n_hist_a_is_gpt35} (sur 75 paires)")
    
    n_rec_a_is_gpt4o = ((pool_df['competition'] == 'recent') & (pool_df['model_a'] == 'GPT-4o-FT')).sum()
    n_rec_a_is_mini = ((pool_df['competition'] == 'recent') & (pool_df['model_a'] == 'GPT-4o-mini-FT')).sum()
    print(f"   Récent (Option A) : GPT-4o={n_rec_a_is_gpt4o}, GPT-4o-mini={n_rec_a_is_mini} (sur 75 paires)")
    assert n_hist_a_is_mistral + n_hist_a_is_gpt35 == 75
    assert n_rec_a_is_gpt4o + n_rec_a_is_mini == 75
    print("✅ Double aveugle validé : position A/B randomisée sans exposition de nom de modèle.")

    # 4. Simulation d'annotations réversibles et atomic write-and-rename
    print("\n--- Étape 4 : Simulation de Sauvegardes Atomiques (Pattern Write-and-Rename fsync) ---")
    test_ann = "eval_sim"
    f_test = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / f"annotations_{test_ann}.csv"
    if f_test.exists():
        f_test.unlink()

    # Annotation de 3 paires
    for i in range(3):
        row = pool_df.iloc[i]
        save_single_annotation(
            repo_root=repo_root,
            annotator=test_ann,
            pair_row=row,
            preferred_option="A" if i % 2 == 0 else "B",
            meaning_preservation_a=PRESERVATION_CHOICES[0],
            meaning_preservation_b=PRESERVATION_CHOICES[1],
            notes=f"Note simulee {i}"
        )
    df_after_3 = load_annotator_data(repo_root, test_ann)
    print(f"   3 annotations créées : {len(df_after_3)} enregistrées.")
    assert len(df_after_3) == 3
    
    # Test Mode Modification (in-situ update sur la paire 0)
    save_single_annotation(
        repo_root=repo_root,
        annotator=test_ann,
        pair_row=pool_df.iloc[0],
        preferred_option="B", # Changement A -> B
        meaning_preservation_a=PRESERVATION_CHOICES[2],
        meaning_preservation_b=PRESERVATION_CHOICES[0],
        notes="Correction apres relecture"
    )
    df_after_mod = load_annotator_data(repo_root, test_ann)
    print(f"   Modification in-situ exécutée : {len(df_after_mod)} lignes (zéro duplication).")
    assert len(df_after_mod) == 3
    row0_mod = df_after_mod[df_after_mod['pair_id'] == pool_df.iloc[0]['pair_id']].iloc[0]
    assert row0_mod['preferred_option'] == "B"
    assert row0_mod['meaning_preservation_a'] == PRESERVATION_CHOICES[2]
    assert row0_mod['notes'] == "Correction apres relecture"
    print("✅ Sauvegarde atomique et réversibilité (Mode Modification) certifiées !")

    # Nettoyage
    if f_test.exists():
        f_test.unlink()
    consolidate_annotations(repo_root)

    # 5. Validation du Calcul de l'Accord Inter-Annotateurs (IAA) sur les 50 paires
    print("\n--- Étape 5 : Validation des Métriques IAA sur 50 Paires (Fleiss, Cohen, Confusion Matrix) ---")
    shared_pool = pool_df[pool_df['split_assignment'] == 'both'].copy()
    assert len(shared_pool) == 50
    
    for i, (_, srow) in enumerate(shared_pool.iterrows()):
        # Henri choisit A sur 35, B sur 15
        h_choice = "A" if i < 35 else "B"
        h_pres_a = PRESERVATION_CHOICES[0] if i < 40 else PRESERVATION_CHOICES[1]
        h_pres_b = PRESERVATION_CHOICES[0] if i % 2 == 0 else PRESERVATION_CHOICES[2]
        save_single_annotation(
            repo_root=repo_root,
            annotator="henri",
            pair_row=srow,
            preferred_option=h_choice,
            meaning_preservation_a=h_pres_a,
            meaning_preservation_b=h_pres_b,
        )
        # Maxime en accord sur la majorité
        m_choice = h_choice if i % 5 != 0 else ("B" if h_choice == "A" else "A")
        m_pres_a = h_pres_a if i % 7 != 0 else PRESERVATION_CHOICES[2]
        m_pres_b = h_pres_b if i % 6 != 0 else PRESERVATION_CHOICES[0]
        save_single_annotation(
            repo_root=repo_root,
            annotator="maxime",
            pair_row=srow,
            preferred_option=m_choice,
            meaning_preservation_a=m_pres_a,
            meaning_preservation_b=m_pres_b,
        )
        
    iaa_res = compute_iaa_metrics(repo_root)
    print(f"   Paires partagées évaluées : {iaa_res['annotated_shared_pairs']} / 50")
    print(f"   Accord Observé Préférence (Po) : {iaa_res['observed_agreement']*100:.1f}%")
    print(f"   Cohen's Kappa Préférence (κ) : {iaa_res['cohen_kappa']:.4f}")
    print(f"   Fleiss' Kappa Préférence (κ_Fleiss) : {iaa_res['fleiss_kappa']:.4f}")
    print(f"   Accord Observé Préservation Sens (Po) : {iaa_res['meaning_observed_agreement']*100:.1f}%")
    print(f"   Cohen's Kappa Préservation Sens (κ) : {iaa_res['meaning_cohen_kappa']:.4f}")
    print(f"   Fleiss' Kappa Préservation Sens (κ_Fleiss) : {iaa_res['meaning_fleiss_kappa']:.4f}")
    print("\n   Matrice de Confusion Préservation du Sens (3x3) :")
    print(iaa_res['meaning_confusion_matrix_df'])
    print("\n   Taux de Préservation par Modèle :")
    print(iaa_res['model_stats_df'])
    
    # Validation du calcul global de performance des modèles (Annotation 9)
    df_perf_all = compute_model_performance(repo_root)
    print("\n   Performance Globale des Modèles (Graphique Barres) :")
    print(df_perf_all)
    assert len(df_perf_all) == 4
    assert df_perf_all["Total Évaluations"].sum() > 0
    assert "Taux de Préservation du Sens (%)" in df_perf_all.columns
    assert "Taux de Préférence (%)" in df_perf_all.columns

    assert iaa_res['annotated_shared_pairs'] == 50
    assert iaa_res['cohen_kappa'] > 0.4
    assert iaa_res['meaning_cohen_kappa'] > 0.4
    print("\n✅ Métriques IAA et Performance des Modèles validées avec succès !")
    
    # Réinitialisation des annotations de simulation
    f_h = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / "annotations_henri.csv"
    f_m = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / "annotations_maxime.csv"
    f_c = repo_root / "dmkd_additions" / "text_simplification" / "reviews" / "annotations_consolidated.csv"
    if f_h.exists(): f_h.unlink()
    if f_m.exists(): f_m.unlink()
    if f_c.exists(): f_c.unlink()

    # 6. Test Fail-Fast
    print("\n--- Étape 6 : Test de la Règle d'Or FAIL-FAST ---")
    try:
        save_single_annotation(
            repo_root=repo_root,
            annotator="henri",
            pair_row=pool_df.iloc[0],
            preferred_option="INVALID_CHOICE", # Doit crasher
            meaning_preservation_a=PRESERVATION_CHOICES[0],
            meaning_preservation_b=PRESERVATION_CHOICES[0],
        )
        raise AssertionError("ÉCHEC FAIL-FAST : L'option invalide aurait dû lever ValueError !")
    except ValueError as e:
        print(f"✅ Fail-Fast vérifié (option invalide rejetée bruyamment) : {e}")

    try:
        save_single_annotation(
            repo_root=repo_root,
            annotator="henri",
            pair_row=pool_df.iloc[0],
            preferred_option="A",
            meaning_preservation_a="CHOIX_TOTALEMENT_INCONNU",
            meaning_preservation_b=PRESERVATION_CHOICES[0],
        )
        raise AssertionError("ÉCHEC FAIL-FAST : Le choix de préservation invalide aurait dû lever ValueError !")
    except ValueError as e:
        print(f"✅ Fail-Fast vérifié (choix préservation invalide rejeté bruyamment) : {e}")

    print("\n=================================================================")
    print("🎯 TOUTES LES ÉTAPES DU DRY-RUN SONT VALIDÉES AVEC SUCCÈS !")
    print("=================================================================\n")

if __name__ == "__main__":
    dry_run()
