"""Moteur de données, échantillonnage stratifié et calculs IAA

pour l'étude d'annotation humaine DMKD (Chantier 2).
Protocole Split-and-Overlap (150 paires CECRL A2-C2, Henri & Maxime).
Règle d'or : FAIL-FAST (aucune exception masquée, aucun fallback cosmétique).
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

CEFR_LEVELS = ["A2", "B1", "B2", "C1", "C2"]

# Protocole à 3 boutons discrets pour la préservation du sens (Arbitrage Michalis Vlachos)
PRESERVATION_CHOICES = [
    "« Oui » (conservé)",
    "« Non » (non conservé)",
    "« Je ne sais pas / Flou »",
]

# Protocole à 3 boutons discrets pour la question 1 : « Le contenu est-il plus simple ? »
SIMPLER_CHOICES = [
    "« Oui » (plus simple)",
    "« Non » (pas plus simple)",
    "« Je ne sais pas / Flou »",
]

__all__ = [
    "CEFR_LEVELS",
    "PRESERVATION_CHOICES",
    "SIMPLER_CHOICES",
    "normalize_preservation_choice",
    "normalize_simpler_choice",
    "find_repo_root",
    "clean_sentence",
    "load_raw_datasets",
    "generate_annotation_pool",
    "get_or_create_pool",
    "save_atomic_dataframe",
    "get_annotator_filepath",
    "load_annotator_data",
    "save_single_annotation",
    "consolidate_annotations",
    "compute_fleiss_kappa",
    "compute_krippendorff_alpha_ordinal",
    "compute_iaa_metrics",
    "compute_model_performance",
    "generate_latex_human_study_macros",
]


def normalize_preservation_choice(val: Any) -> str:
    """Normalise les choix de préservation du sens de façon Fail-Fast."""
    if val is None or pd.isna(val):
        return PRESERVATION_CHOICES[0]
    s = str(val).strip()
    if s in PRESERVATION_CHOICES:
        return s
    s_lower = s.lower()
    if "oui" in s_lower or "yes" in s_lower:
        return PRESERVATION_CHOICES[0]
    if "non" in s_lower or "no" in s_lower:
        return PRESERVATION_CHOICES[1]
    if "flou" in s_lower or "sais pas" in s_lower or "unclear" in s_lower or "?" in s_lower:
        return PRESERVATION_CHOICES[2]
    # Rétrocompatibilité avec d'anciens scores Likert (1-5)
    try:
        score = int(float(s))
        if score >= 4:
            return PRESERVATION_CHOICES[0]
        elif score <= 2:
            return PRESERVATION_CHOICES[1]
        else:
            return PRESERVATION_CHOICES[2]
    except (ValueError, TypeError):
        pass
    raise ValueError(
        f"Valeur invalide pour la préservation du sens : '{val}'. "
        f"Options valides attendues : {PRESERVATION_CHOICES}"
    )


def normalize_simpler_choice(val: Any) -> str:
    """Normalise les choix pour la question 'Le contenu est-il plus simple ?' de façon Fail-Fast."""
    if val is None or pd.isna(val):
        return SIMPLER_CHOICES[0]
    s = str(val).strip()
    if s in SIMPLER_CHOICES:
        return s
    s_lower = s.lower()
    if "oui" in s_lower or "yes" in s_lower:
        return SIMPLER_CHOICES[0]
    if "non" in s_lower or "no" in s_lower:
        return SIMPLER_CHOICES[1]
    if "flou" in s_lower or "sais pas" in s_lower or "unsure" in s_lower or "unclear" in s_lower or "?" in s_lower:
        return SIMPLER_CHOICES[2]
    raise ValueError(
        f"Valeur invalide pour la simplicité du contenu : '{val}'. "
        f"Options valides attendues : {SIMPLER_CHOICES}"
    )


def find_repo_root(start_dir: Optional[Path] = None) -> Path:
    """Trouve la racine du dépôt git de manière déterministe."""
    if start_dir is None:
        start_dir = Path(__file__).resolve().parent

    curr = start_dir
    while curr.parent != curr:
        if (curr / "results").is_dir() and (curr / "dmkd_additions").is_dir():
            return curr
        curr = curr.parent

    # Si non trouvé, lever une exception explicite (Fail-Fast)
    raise FileNotFoundError(
        f"Impossible de localiser la racine du dépôt depuis {start_dir}. "
        "Les répertoires 'results' et 'dmkd_additions' doivent exister."
    )


def clean_sentence(text: Any) -> str:
    """Nettoie les variations d'apostrophes et d'espaces superflus."""
    if text is None or pd.isna(text):
        return ""
    s = str(text)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def load_raw_datasets(repo_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge et aligne les prédictions historiques et récentes.

    Lève FileNotFoundError ou ValueError si les données sont invalides.
    """
    # 1. Sources historiques
    mistral_raw_path = (
        repo_root
        / "results"
        / "text_simplification"
        / "MistralEvaluation"
        / "predictions_fine_tuned.csv"
    )
    mistral_fmt_path = (
        repo_root
        / "results"
        / "text_simplification"
        / "MistralEvaluation"
        / "predictions_fine_tuned_formatted.csv"
    )
    gpt35_fmt_path = (
        repo_root
        / "results"
        / "text_simplification"
        / "OpenAIEvaluation"
        / "gpt-3.5-turbo-1106-finetuned_formatted_predictions.csv"
    )

    for p in [mistral_raw_path, mistral_fmt_path, gpt35_fmt_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Fichier historique manquant : {p}")

    df_mistral_raw = pd.read_csv(mistral_raw_path)
    df_mistral_fmt = pd.read_csv(mistral_fmt_path)
    df_gpt35_fmt = pd.read_csv(gpt35_fmt_path)

    if len(df_mistral_raw) != 1000 or len(df_mistral_fmt) != 1000:
        raise ValueError(
            f"Taille inattendue pour Mistral : raw={len(df_mistral_raw)}, fmt={len(df_mistral_fmt)} (attendu: 1000)"
        )
    if len(df_gpt35_fmt) != 1000:
        raise ValueError(
            f"Taille inattendue pour GPT-3.5 : {len(df_gpt35_fmt)} (attendu: 1000)"
        )

    # Extraire les niveaux CECRL des prompts Mistral
    levels = []
    for raw_prompt in df_mistral_raw.iloc[:, 0]:
        m = re.search(r"de niveau (A1|A2|B1|B2|C1|C2)", str(raw_prompt))
        if not m:
            raise ValueError(
                f"Impossible d'extraire le niveau CECRL du prompt : {str(raw_prompt)[:100]}"
            )
        levels.append(m.group(1))

    df_mistral_fmt["cefr_level"] = levels
    df_mistral_fmt["clean"] = df_mistral_fmt["Original"].apply(clean_sentence)
    df_gpt35_fmt["clean"] = df_gpt35_fmt["Original"].apply(clean_sentence)

    # Déduplication et fusion sur 'clean'
    df_mistral_dedup = df_mistral_fmt.drop_duplicates(
        subset=["clean"], keep="first"
    )
    df_gpt35_dedup = df_gpt35_fmt.drop_duplicates(
        subset=["clean"], keep="first"
    )

    df_hist = pd.merge(
        df_mistral_dedup[["clean", "Original", "cefr_level", "Simplified"]].rename(
            columns={
                "Original": "original_sentence",
                "Simplified": "simplified_mistral",
            }
        ),
        df_gpt35_dedup[["clean", "Simplified"]].rename(
            columns={"Simplified": "simplified_gpt35"}
        ),
        on="clean",
        how="inner",
    )

    # 2. Sources récentes
    gpt4o_path = (
        repo_root
        / "results"
        / "dmkd_additions"
        / "text_simplification"
        / "gpt_evaluation"
        / "test_set_ft_gpt-4o-2024-08-06_university-of-lausanne__AYDSOp5f_predictions.csv"
    )
    gpt4o_mini_path = (
        repo_root
        / "results"
        / "dmkd_additions"
        / "text_simplification"
        / "gpt_evaluation"
        / "test_set_ft_gpt-4o-mini-2024-07-18_university-of-lausanne__AYDK3AI2_predictions.csv"
    )

    for p in [gpt4o_path, gpt4o_mini_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Fichier récent manquant : {p}")

    df_gpt4o = pd.read_csv(gpt4o_path)
    df_gpt4o_mini = pd.read_csv(gpt4o_mini_path)

    if len(df_gpt4o) != 500 or len(df_gpt4o_mini) != 500:
        raise ValueError(
            f"Taille inattendue : gpt4o={len(df_gpt4o)}, gpt4o_mini={len(df_gpt4o_mini)} (attendu: 500)"
        )

    # Vérification d'alignement phrase par phrase
    if not (df_gpt4o["Original"] == df_gpt4o_mini["Original"]).all():
        raise ValueError(
            "Désalignement détecté entre les phrases originales de GPT-4o et GPT-4o-mini !"
        )

    df_recent = pd.DataFrame(
        {
            "original_sentence": df_gpt4o["Original"],
            "cefr_level": df_gpt4o["Difficulty"],
            "simplified_gpt4o": df_gpt4o["simplified"],
            "simplified_gpt4o_mini": df_gpt4o_mini["simplified"],
        }
    )

    return df_hist, df_recent


def generate_annotation_pool(
    repo_root: Path, seed: int = 42, pool_path: Optional[Path] = None
) -> pd.DataFrame:
    """Génère le pool d'annotation stratifié de 150 paires selon le protocole split-and-overlap.

    - 30 paires par niveau (A2, B1, B2, C1, C2) = 150 paires.
    - Par niveau : 15 paires Historiques (Mistral FT vs GPT-3.5 FT) + 15 paires Récentes (GPT-4o FT vs GPT-4o-mini FT).
    - Par groupe de 15 paires :
        * 3 paires 'both' (partagées double aveugle) -> total 30 partagées.
        * 6 paires 'henri' -> total 60 réparties.
        * 6 paires 'maxime' -> total 60 réparties.
    - Double aveugle : ordre de présentation randomisé 50/50 (Model A vs Model B) pour neutraliser le biais de position.
    """
    df_hist, df_recent = load_raw_datasets(repo_root)

    rng = np.random.default_rng(seed)
    pairs_list: List[Dict[str, Any]] = []

    global_pair_idx = 1

    for level in CEFR_LEVELS:
        # 1. Sélection Historique : 15 paires
        sub_hist = df_hist[df_hist["cefr_level"] == level]
        if len(sub_hist) < 15:
            raise ValueError(
                f"Nombre insuffisant de paires historiques pour le niveau {level} : {len(sub_hist)} disponibles, 15 requises."
            )
        sampled_hist_indices = rng.choice(
            sub_hist.index, size=15, replace=False
        )
        sampled_hist = sub_hist.loc[sampled_hist_indices].copy()

        # Attribution des rôles : 5 'both', 5 'henri', 5 'maxime' (total: 15)
        roles = ["both"] * 5 + ["henri"] * 5 + ["maxime"] * 5
        rng.shuffle(roles)
        sampled_hist["split_assignment"] = roles

        for _, row in sampled_hist.iterrows():
            # Randomisation A/B
            flip = rng.random() < 0.5
            if flip:
                mod_a, mod_b = "Mistral-7B-FT", "GPT-3.5-turbo-FT"
                simp_a, simp_b = (
                    row["simplified_mistral"],
                    row["simplified_gpt35"],
                )
            else:
                mod_a, mod_b = "GPT-3.5-turbo-FT", "Mistral-7B-FT"
                simp_a, simp_b = (
                    row["simplified_gpt35"],
                    row["simplified_mistral"],
                )

            pairs_list.append(
                {
                    "pair_id": f"P{global_pair_idx:03d}",
                    "cefr_level": level,
                    "competition": "historical",
                    "model_pair_type": "Mistral-7B-FT vs GPT-3.5-turbo-FT",
                    "split_assignment": row["split_assignment"],
                    "original_sentence": row["original_sentence"],
                    "model_a": mod_a,
                    "model_b": mod_b,
                    "simplification_a": simp_a,
                    "simplification_b": simp_b,
                }
            )
            global_pair_idx += 1

        # 2. Sélection Récente : 15 paires
        sub_recent = df_recent[df_recent["cefr_level"] == level]
        if len(sub_recent) < 15:
            raise ValueError(
                f"Nombre insuffisant de paires récentes pour le niveau {level} : {len(sub_recent)} disponibles, 15 requises."
            )
        sampled_recent_indices = rng.choice(
            sub_recent.index, size=15, replace=False
        )
        sampled_recent = sub_recent.loc[sampled_recent_indices].copy()

        roles_rec = ["both"] * 5 + ["henri"] * 5 + ["maxime"] * 5
        rng.shuffle(roles_rec)
        sampled_recent["split_assignment"] = roles_rec

        for _, row in sampled_recent.iterrows():
            flip = rng.random() < 0.5
            if flip:
                mod_a, mod_b = "GPT-4o-FT", "GPT-4o-mini-FT"
                simp_a, simp_b = (
                    row["simplified_gpt4o"],
                    row["simplified_gpt4o_mini"],
                )
            else:
                mod_a, mod_b = "GPT-4o-mini-FT", "GPT-4o-FT"
                simp_a, simp_b = (
                    row["simplified_gpt4o_mini"],
                    row["simplified_gpt4o"],
                )

            pairs_list.append(
                {
                    "pair_id": f"P{global_pair_idx:03d}",
                    "cefr_level": level,
                    "competition": "recent",
                    "model_pair_type": "GPT-4o-FT vs GPT-4o-mini-FT",
                    "split_assignment": row["split_assignment"],
                    "original_sentence": row["original_sentence"],
                    "model_a": mod_a,
                    "model_b": mod_b,
                    "simplification_a": simp_a,
                    "simplification_b": simp_b,
                }
            )
            global_pair_idx += 1

    pool_df = pd.DataFrame(pairs_list)

    # Validation stricte des invariants (Fail-Fast)
    if len(pool_df) != 150:
        raise ValueError(
            f"Le pool généré compte {len(pool_df)} paires (attendu: 150)."
        )

    for level in CEFR_LEVELS:
        cnt = (pool_df["cefr_level"] == level).sum()
        if cnt != 30:
            raise ValueError(
                f"Incohérence stratification : niveau {level} a {cnt} paires (attendu: 30)."
            )

    shared_cnt = (pool_df["split_assignment"] == "both").sum()
    henri_cnt = (pool_df["split_assignment"] == "henri").sum()
    maxime_cnt = (pool_df["split_assignment"] == "maxime").sum()

    if shared_cnt != 50 or henri_cnt != 50 or maxime_cnt != 50:
        raise ValueError(
            f"Incohérence split-and-overlap : both={shared_cnt}, henri={henri_cnt}, maxime={maxime_cnt} (attendu: 50, 50, 50)."
        )

    if pool_path is not None:
        save_atomic_dataframe(pool_df, pool_path)

    return pool_df


def get_or_create_pool(repo_root: Path) -> pd.DataFrame:
    """Charge le pool d'annotation s'il existe, ou le génère et le sauvegarde de façon atomique."""
    reviews_dir = (
        repo_root
        / "dmkd_additions"
        / "text_simplification"
        / "reviews"
    )
    reviews_dir.mkdir(parents=True, exist_ok=True)
    pool_path = reviews_dir / "annotation_pool_150.csv"

    if pool_path.is_file():
        df = pd.read_csv(pool_path)
        if len(df) == 150:
            return df
        # Si taille anormale, regénérer bruyamment
        raise ValueError(
            f"Fichier de pool corrompu ou incomplet ({len(df)} lignes) : {pool_path}"
        )

    return generate_annotation_pool(repo_root, seed=42, pool_path=pool_path)


def save_atomic_dataframe(df: pd.DataFrame, target_path: Path) -> None:
    """Sauvegarde atomique anti-corruption selon le pattern Write-and-Rename.

    1. Écriture sur un fichier temporaire .tmp
    2. Flush applicatif f.flush()
    3. Verrouillage physique os.fsync(f.fileno())
    4. Substitution atomique os.replace() avec retries sous Windows
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())

    # Sous Windows, un watcher (ex: Streamlit) ou un scanner peut verrouiller brièvement le fichier
    for attempt in range(10):
        try:
            os.replace(tmp_path, target_path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            import time
            time.sleep(0.05)


def get_annotator_filepath(repo_root: Path, annotator: str) -> Path:
    """Retourne le chemin canonique du fichier d'annotations d'un annotateur."""
    annotator_clean = annotator.strip().lower()
    reviews_dir = (
        repo_root
        / "dmkd_additions"
        / "text_simplification"
        / "reviews"
    )
    reviews_dir.mkdir(parents=True, exist_ok=True)
    return reviews_dir / f"annotations_{annotator_clean}.csv"


def load_annotator_data(repo_root: Path, annotator: str) -> pd.DataFrame:
    """Charge les annotations existantes pour un annotateur donné."""
    filepath = get_annotator_filepath(repo_root, annotator)
    canonical_columns = [
        "pair_id",
        "annotator",
        "timestamp",
        "preferred_option",  # 'A' ou 'B'
        "preferred_model",  # nom du modèle choisi
        "content_simpler_a",  # « Oui » (plus simple), « Non » (pas plus simple), « Je ne sais pas / Flou »
        "content_simpler_b",  # « Oui » (plus simple), « Non » (pas plus simple), « Je ne sais pas / Flou »
        "meaning_preservation_a",  # « Oui » (conservé), « Non » (non conservé), « Je ne sais pas / Flou »
        "meaning_preservation_b",  # « Oui » (conservé), « Non » (non conservé), « Je ne sais pas / Flou »
        "notes",
    ]
    if not filepath.is_file():
        # Retourne un DataFrame vide avec les colonnes requises typées object
        return pd.DataFrame(
            {col: pd.Series(dtype="object") for col in canonical_columns}
        )
    df = pd.read_csv(filepath)
    # Rétrocompatibilité : ajouter content_simpler_a / content_simpler_b si manquantes
    if "content_simpler_a" not in df.columns:
        df["content_simpler_a"] = SIMPLER_CHOICES[0]
    if "content_simpler_b" not in df.columns:
        df["content_simpler_b"] = SIMPLER_CHOICES[0]
    return df


def save_single_annotation(
    repo_root: Path,
    annotator: str,
    pair_row: pd.Series,
    preferred_option: str,
    content_simpler_a: str = SIMPLER_CHOICES[0],
    content_simpler_b: str = SIMPLER_CHOICES[0],
    meaning_preservation_a: Optional[Any] = None,
    meaning_preservation_b: Optional[Any] = None,
    notes: str = "",
    # Paramètres de rétrocompatibilité éventuelle
    fidelity_score_a: Optional[Any] = None,
    fidelity_score_b: Optional[Any] = None,
    fluency_score_a: Optional[Any] = None,
    fluency_score_b: Optional[Any] = None,
) -> pd.DataFrame:
    """Enregistre ou met à jour une annotation de manière atomique (in-situ update / append)."""
    if preferred_option not in ["A", "B"]:
        raise ValueError(
            f"Option préférée invalide : {preferred_option} (attendu 'A' ou 'B')"
        )

    norm_simpler_a = normalize_simpler_choice(content_simpler_a)
    norm_simpler_b = normalize_simpler_choice(content_simpler_b)

    val_a = meaning_preservation_a if meaning_preservation_a is not None else fidelity_score_a
    val_b = meaning_preservation_b if meaning_preservation_b is not None else fidelity_score_b

    norm_a = normalize_preservation_choice(val_a)
    norm_b = normalize_preservation_choice(val_b)

    preferred_model = (
        pair_row["model_a"] if preferred_option == "A" else pair_row["model_b"]
    )
    pair_id = pair_row["pair_id"]

    df = load_annotator_data(repo_root, annotator)

    new_record = {
        "pair_id": pair_id,
        "annotator": annotator.strip().lower(),
        "timestamp": datetime.now().isoformat(),
        "preferred_option": preferred_option,
        "preferred_model": preferred_model,
        "content_simpler_a": norm_simpler_a,
        "content_simpler_b": norm_simpler_b,
        "meaning_preservation_a": norm_a,
        "meaning_preservation_b": norm_b,
        "notes": str(notes).strip() if notes is not None else "",
    }

    if not df.empty and pair_id in df["pair_id"].values:
        # Mise à jour in-situ
        idx = df[df["pair_id"] == pair_id].index[0]
        for k, v in new_record.items():
            df.loc[idx, k] = v
    else:
        # Ajout
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

    filepath = get_annotator_filepath(repo_root, annotator)
    save_atomic_dataframe(df, filepath)
    return df


def consolidate_annotations(
    repo_root: Path, persist: bool = False
) -> pd.DataFrame:
    """Agrège toutes les annotations individuelles dans un fichier consolidé unique."""
    reviews_dir = (
        repo_root
        / "dmkd_additions"
        / "text_simplification"
        / "reviews"
    )
    consolidated_path = reviews_dir / "annotations_consolidated.csv"

    frames = []
    for f in reviews_dir.glob("annotations_*.csv"):
        if f.name in [
            "annotations_consolidated.csv",
            "annotation_pool_150.csv",
        ]:
            continue
        try:
            sub_df = pd.read_csv(f)
            if not sub_df.empty:
                frames.append(sub_df)
        except Exception as e:
            raise RuntimeError(
                f"Échec de lecture lors de la consolidation de {f} : {e}"
            )

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        if "content_simpler_a" not in merged.columns:
            merged["content_simpler_a"] = SIMPLER_CHOICES[0]
        if "content_simpler_b" not in merged.columns:
            merged["content_simpler_b"] = SIMPLER_CHOICES[0]
    else:
        merged = pd.DataFrame(
            columns=[
                "pair_id",
                "annotator",
                "timestamp",
                "preferred_option",
                "preferred_model",
                "content_simpler_a",
                "content_simpler_b",
                "meaning_preservation_a",
                "meaning_preservation_b",
                "notes",
            ]
        )

    if persist:
        save_atomic_dataframe(merged, consolidated_path)
    return merged


def compute_fleiss_kappa(categories_counts: np.ndarray) -> float:
    """Calcule le coefficient Fleiss' Kappa exact selon la formulation standard (Fleiss, 1971).

    categories_counts : matrice 2D (N_sujets x k_categories), où chaque case [i, j]
    contient le nombre d'évaluateurs ayant classé le sujet i dans la catégorie j.
    """
    N, k = categories_counts.shape
    if N == 0 or k == 0:
        return 0.0

    n_raters_per_subject = categories_counts.sum(axis=1)
    m = n_raters_per_subject[0]
    if not (n_raters_per_subject == m).all():
        raise ValueError(
            "Fleiss' Kappa requiert un nombre constant d'évaluateurs par sujet."
        )

    if m <= 1:
        return 0.0

    # Probabilité marginale par catégorie p_j
    total_ratings = N * m
    p_j = categories_counts.sum(axis=0) / total_ratings

    # Accord observé par sujet P_i
    P_i = (
        (categories_counts**2).sum(axis=1) - m
    ) / (m * (m - 1))
    P_bar = P_i.mean()

    # Accord aléatoire attendu P_e_bar
    P_e_bar = (p_j**2).sum()

    if np.isclose(1.0 - P_e_bar, 0.0):
        return 1.0

    kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
    return float(kappa)



def compute_krippendorff_alpha_ordinal(
    ratings_h: List[int], ratings_m: List[int]
) -> float:
    """Calcule le coefficient Alpha de Krippendorff ordinal (m=2 observateurs).

    ratings_h, ratings_m : listes de scores discrets ordonnés (ex: 0, 1, 2)
    associés aux mêmes n unités d'évaluation.
    """
    if len(ratings_h) != len(ratings_m):
        raise ValueError("ratings_h et ratings_m doivent avoir la même longueur.")
    n = len(ratings_h)
    if n == 0:
        return 1.0

    all_vals = list(ratings_h) + list(ratings_m)
    if len(set(all_vals)) <= 1:
        return 1.0

    if ratings_h == ratings_m:
        return 1.0

    categories = sorted(list(set(all_vals)))
    counts = {c: all_vals.count(c) for c in categories}

    # Distance ordinale de Krippendorff : delta_ord(c, k) = sum_{g=min(c,k)}^{max(c,k)} n_g - (n_c + n_k)/2
    delta = {}
    for c in categories:
        for k in categories:
            if c == k:
                delta[(c, k)] = 0.0
            else:
                c_min, c_max = min(c, k), max(c, k)
                in_between = [g for g in categories if c_min <= g <= c_max]
                sum_ng = sum(counts[g] for g in in_between)
                d_val = sum_ng - (counts[c] + counts[k]) / 2.0
                delta[(c, k)] = float(d_val)

    # Désaccord observé D_o
    d_o = sum((delta[(h, m)]) ** 2 for h, m in zip(ratings_h, ratings_m)) / float(n)

    # Désaccord attendu D_e
    total_obs = 2 * n
    denom_de = float(total_obs * (total_obs - 1))
    sum_de = 0.0
    for c in categories:
        for k in categories:
            sum_de += counts[c] * counts[k] * ((delta[(c, k)]) ** 2)
    d_e = sum_de / denom_de

    if np.isclose(d_e, 0.0):
        return 1.0 if np.isclose(d_o, 0.0) else 0.0

    alpha = 1.0 - (d_o / d_e)
    return float(alpha)

def compute_iaa_metrics(repo_root: Path) -> Dict[str, Any]:
    """Calcule toutes les métriques d'accord inter-annotateurs (Cohen's Kappa,

    Fleiss' Kappa, concordance observée, matrices de confusion) sur les 30 paires partagées.
    """
    pool_df = get_or_create_pool(repo_root)
    shared_pool = pool_df[pool_df["split_assignment"] == "both"].copy()

    df_henri = load_annotator_data(repo_root, "henri")
    df_maxime = load_annotator_data(repo_root, "maxime")

    # Jointure sur les paires partagées
    shared_pair_ids = set(shared_pool["pair_id"])
    henri_shared = df_henri[df_henri["pair_id"].isin(shared_pair_ids)]
    maxime_shared = df_maxime[df_maxime["pair_id"].isin(shared_pair_ids)]

    common_annotated_ids = set(henri_shared["pair_id"]).intersection(
        set(maxime_shared["pair_id"])
    )

    total_shared = len(shared_pool)
    n_common = len(common_annotated_ids)

    result: Dict[str, Any] = {
        "total_shared_pairs": total_shared,
        "annotated_shared_pairs": n_common,
        "henri_total_annotated": len(df_henri),
        "maxime_total_annotated": len(df_maxime),
        "henri_shared_annotated": len(henri_shared),
        "maxime_shared_annotated": len(maxime_shared),
    }

    if n_common == 0:
        result["status"] = "no_overlap_annotated"
        result["krippendorff_alpha"] = None
        result["cohen_kappa_pref"] = None
        return result

    # Filtrer et aligner dans le même ordre de pair_id
    sorted_ids = sorted(list(common_annotated_ids))
    sub_h = (
        henri_shared.set_index("pair_id")
        .loc[sorted_ids]
        .reset_index()
    )
    sub_m = (
        maxime_shared.set_index("pair_id")
        .loc[sorted_ids]
        .reset_index()
    )
    sub_pool = (
        shared_pool.set_index("pair_id")
        .loc[sorted_ids]
        .reset_index()
    )

    # 1. Préférence de modèle (Choix Forcé)
    # y_h et y_m contiennent le modèle préféré
    y_h = sub_h["preferred_model"].tolist()
    y_m = sub_m["preferred_model"].tolist()

    # Accord brut
    agreement_count = sum(1 for a, b in zip(y_h, y_m) if a == b)
    observed_agreement = agreement_count / n_common

    # Cohen's Kappa
    # Récupérer l'ensemble de tous les modèles possibles dans ce sous-ensemble
    all_categories = sorted(list(set(y_h).union(set(y_m))))
    if len(all_categories) > 1:
        cohen_k = float(
            cohen_kappa_score(y_h, y_m, labels=all_categories)
        )
    else:
        cohen_k = 1.0 if agreement_count == n_common else 0.0

    # Fleiss' Kappa (pour 2 raters)
    # Construction de la matrice counts (N x k)
    cat_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
    counts_matrix = np.zeros((n_common, len(all_categories)), dtype=int)
    for row_i, (ch, cm) in enumerate(zip(y_h, y_m)):
        counts_matrix[row_i, cat_to_idx[ch]] += 1
        counts_matrix[row_i, cat_to_idx[cm]] += 1

    fleiss_k = compute_fleiss_kappa(counts_matrix)

    # Matrice de confusion
    conf_mat = confusion_matrix(y_h, y_m, labels=all_categories)
    conf_df = pd.DataFrame(
        conf_mat,
        index=[f"Henri: {c}" for c in all_categories],
        columns=[f"Maxime: {c}" for c in all_categories],
    )

    # 2. Accord sur la Simplicité du Contenu (Question 1 : « Le contenu est-il plus simple ? »)
    h_simpler_ratings = []
    m_simpler_ratings = []

    for i in range(n_common):
        h_s_a = normalize_simpler_choice(sub_h.iloc[i].get("content_simpler_a"))
        h_s_b = normalize_simpler_choice(sub_h.iloc[i].get("content_simpler_b"))
        m_s_a = normalize_simpler_choice(sub_m.iloc[i].get("content_simpler_a"))
        m_s_b = normalize_simpler_choice(sub_m.iloc[i].get("content_simpler_b"))

        h_simpler_ratings.extend([h_s_a, h_s_b])
        m_simpler_ratings.extend([m_s_a, m_s_b])

    n_simpler_ratings = len(h_simpler_ratings)
    simpler_agree_cnt = sum(
        1 for a, b in zip(h_simpler_ratings, m_simpler_ratings) if a == b
    )
    simpler_observed_agreement = (
        simpler_agree_cnt / n_simpler_ratings if n_simpler_ratings > 0 else 0.0
    )

    simpler_cohen_k = float(
        cohen_kappa_score(
            h_simpler_ratings, m_simpler_ratings, labels=SIMPLER_CHOICES
        )
    )

    # Fleiss' Kappa pour la simplicité (matrice N_ratings x 3)
    simpler_to_idx = {cat: idx for idx, cat in enumerate(SIMPLER_CHOICES)}
    counts_matrix_simpler = np.zeros(
        (n_simpler_ratings, len(SIMPLER_CHOICES)), dtype=int
    )
    for row_i, (ch, cm) in enumerate(zip(h_simpler_ratings, m_simpler_ratings)):
        counts_matrix_simpler[row_i, simpler_to_idx[ch]] += 1
        counts_matrix_simpler[row_i, simpler_to_idx[cm]] += 1

    simpler_fleiss_k = compute_fleiss_kappa(counts_matrix_simpler)

    # Matrice de confusion 3x3 sur la simplicité
    conf_mat_simpler = confusion_matrix(
        h_simpler_ratings, m_simpler_ratings, labels=SIMPLER_CHOICES
    )
    conf_df_simpler = pd.DataFrame(
        conf_mat_simpler,
        index=[f"Henri : {c}" for c in SIMPLER_CHOICES],
        columns=[f"Maxime : {c}" for c in SIMPLER_CHOICES],
    )

    # 3. Accord sur la Préservation du Sens (Question 2 : « Le sens est-il conservé ? »)
    # Pour chaque paire partagée, on compare l'évaluation de l'Option A et de l'Option B
    h_pres_ratings = []
    m_pres_ratings = []
    model_evaluations: Dict[str, List[str]] = {}

    for i in range(n_common):
        # Récupération et normalisation des choix (avec support des anciennes colonnes si présentes)
        h_a = normalize_preservation_choice(
            sub_h.iloc[i].get("meaning_preservation_a", sub_h.iloc[i].get("fidelity_score_a"))
        )
        h_b = normalize_preservation_choice(
            sub_h.iloc[i].get("meaning_preservation_b", sub_h.iloc[i].get("fidelity_score_b"))
        )
        m_a = normalize_preservation_choice(
            sub_m.iloc[i].get("meaning_preservation_a", sub_m.iloc[i].get("fidelity_score_a"))
        )
        m_b = normalize_preservation_choice(
            sub_m.iloc[i].get("meaning_preservation_b", sub_m.iloc[i].get("fidelity_score_b"))
        )

        h_pres_ratings.extend([h_a, h_b])
        m_pres_ratings.extend([m_a, m_b])

        # Suivi par modèle
        mod_a = sub_pool.iloc[i]["model_a"]
        mod_b = sub_pool.iloc[i]["model_b"]

        model_evaluations.setdefault(mod_a, []).extend([h_a, m_a])
        model_evaluations.setdefault(mod_b, []).extend([h_b, m_b])

    n_ratings = len(h_pres_ratings)
    meaning_agree_cnt = sum(
        1 for a, b in zip(h_pres_ratings, m_pres_ratings) if a == b
    )
    meaning_observed_agreement = (
        meaning_agree_cnt / n_ratings if n_ratings > 0 else 0.0
    )

    meaning_cohen_k = float(
        cohen_kappa_score(
            h_pres_ratings, m_pres_ratings, labels=PRESERVATION_CHOICES
        )
    )

    # Fleiss' Kappa pour la préservation du sens (matrice N_ratings x 3)
    pres_to_idx = {cat: idx for idx, cat in enumerate(PRESERVATION_CHOICES)}
    counts_matrix_meaning = np.zeros(
        (n_ratings, len(PRESERVATION_CHOICES)), dtype=int
    )
    for row_i, (ch, cm) in enumerate(zip(h_pres_ratings, m_pres_ratings)):
        counts_matrix_meaning[row_i, pres_to_idx[ch]] += 1
        counts_matrix_meaning[row_i, pres_to_idx[cm]] += 1

    meaning_fleiss_k = compute_fleiss_kappa(counts_matrix_meaning)

    # Matrice de confusion 3x3 sur la préservation du sens
    conf_mat_meaning = confusion_matrix(
        h_pres_ratings, m_pres_ratings, labels=PRESERVATION_CHOICES
    )
    conf_df_meaning = pd.DataFrame(
        conf_mat_meaning,
        index=[f"Henri : {c}" for c in PRESERVATION_CHOICES],
        columns=[f"Maxime : {c}" for c in PRESERVATION_CHOICES],
    )

    # 4. Accord Ordinal Global (Krippendorff's Alpha sur simplicité et sens combinés)
    def _to_ordinal_score(val_str: Any) -> int:
        s = str(val_str).lower()
        if "oui" in s or "yes" in s:
            return 2
        if "flou" in s or "sais pas" in s or "unclear" in s or "unsure" in s or "?" in s:
            return 1
        if "non" in s or "no" in s:
            return 0
        try:
            f = float(s)
            if f >= 4.0:
                return 2
            elif f <= 2.0:
                return 0
            return 1
        except (ValueError, TypeError):
            return 1

    h_ord_all: List[int] = []
    m_ord_all: List[int] = []
    for i in range(n_common):
        h_s_a = normalize_simpler_choice(sub_h.iloc[i].get("content_simpler_a"))
        h_s_b = normalize_simpler_choice(sub_h.iloc[i].get("content_simpler_b"))
        m_s_a = normalize_simpler_choice(sub_m.iloc[i].get("content_simpler_a"))
        m_s_b = normalize_simpler_choice(sub_m.iloc[i].get("content_simpler_b"))
        h_m_a = normalize_preservation_choice(
            sub_h.iloc[i].get("meaning_preservation_a", sub_h.iloc[i].get("fidelity_score_a"))
        )
        h_m_b = normalize_preservation_choice(
            sub_h.iloc[i].get("meaning_preservation_b", sub_h.iloc[i].get("fidelity_score_b"))
        )
        m_m_a = normalize_preservation_choice(
            sub_m.iloc[i].get("meaning_preservation_a", sub_m.iloc[i].get("fidelity_score_a"))
        )
        m_m_b = normalize_preservation_choice(
            sub_m.iloc[i].get("meaning_preservation_b", sub_m.iloc[i].get("fidelity_score_b"))
        )
        h_ord_all.extend([_to_ordinal_score(h_s_a), _to_ordinal_score(h_s_b), _to_ordinal_score(h_m_a), _to_ordinal_score(h_m_b)])
        m_ord_all.extend([_to_ordinal_score(m_s_a), _to_ordinal_score(m_s_b), _to_ordinal_score(m_m_a), _to_ordinal_score(m_m_b)])

    krip_alpha = compute_krippendorff_alpha_ordinal(h_ord_all, m_ord_all)

    # Statistiques par modèle (% Oui, % Non, % Flou)
    model_stats = []
    for mod_name, ratings in model_evaluations.items():
        tot = len(ratings)
        n_oui = sum(1 for r in ratings if r == PRESERVATION_CHOICES[0])
        n_non = sum(1 for r in ratings if r == PRESERVATION_CHOICES[1])
        n_flou = sum(1 for r in ratings if r == PRESERVATION_CHOICES[2])
        model_stats.append(
            {
                "Modèle": mod_name,
                "Évaluations": tot,
                "Sens Conservé (%)": f"{(n_oui / tot)*100:.1f}%" if tot > 0 else "N/A",
                "Non Conservé (%)": f"{(n_non / tot)*100:.1f}%" if tot > 0 else "N/A",
                "Flou / Incertain (%)": f"{(n_flou / tot)*100:.1f}%" if tot > 0 else "N/A",
            }
        )
    df_model_stats = pd.DataFrame(model_stats)

    result.update(
        {
            "status": "computed",
            # Accord Préférence Globale
            "observed_agreement": observed_agreement,
            "cohen_kappa": cohen_k,
            "cohen_kappa_pref": cohen_k,
            "krippendorff_alpha": round(krip_alpha, 4),
            "fleiss_kappa": fleiss_k,
            "categories": all_categories,
            "confusion_matrix_df": conf_df,
            # Accord Simplicité du Contenu (Question 1)
            "simpler_agreement": simpler_observed_agreement,
            "simpler_observed_agreement": simpler_observed_agreement,
            "simpler_cohen_kappa": simpler_cohen_k,
            "simpler_fleiss_kappa": simpler_fleiss_k,
            "simpler_confusion_matrix": conf_df_simpler,
            "simpler_confusion_matrix_df": conf_df_simpler,
            # Accord Préservation du Sens (Question 2)
            "meaning_observed_agreement": meaning_observed_agreement,
            "meaning_cohen_kappa": meaning_cohen_k,
            "meaning_fleiss_kappa": meaning_fleiss_k,
            "meaning_confusion_matrix_df": conf_df_meaning,
            "model_stats_df": df_model_stats,
        }
    )

    return result


def compute_model_performance(repo_root: Optional[Path] = None) -> pd.DataFrame:
    """Calcule les taux de préservation du sens, de simplicité et de préférence par modèle."""
    if repo_root is None:
        repo_root = find_repo_root()
    else:
        repo_root = Path(repo_root)

    pool_df = get_or_create_pool(repo_root)
    df_ann = consolidate_annotations(repo_root)

    models = [
        "Mistral-7B-FT",
        "GPT-3.5-turbo-FT",
        "GPT-4o-FT",
        "GPT-4o-mini-FT",
    ]

    stats = {
        m: {
            "eval_count": 0,
            "simpler_oui_count": 0,
            "sens_oui_count": 0,
            "pref_count": 0,
        }
        for m in models
    }

    if not df_ann.empty:
        pool_map = pool_df.set_index("pair_id")[
            ["model_a", "model_b"]
        ].to_dict(orient="index")

        for _, row in df_ann.iterrows():
            pid = row["pair_id"]
            if pid not in pool_map:
                continue
            mod_a = pool_map[pid]["model_a"]
            mod_b = pool_map[pid]["model_b"]
            pref_opt = str(row.get("preferred_option", "")).strip().upper()
            simpler_a = normalize_simpler_choice(
                row.get("content_simpler_a")
            )
            simpler_b = normalize_simpler_choice(
                row.get("content_simpler_b")
            )
            pres_a = normalize_preservation_choice(
                row.get("meaning_preservation_a")
            )
            pres_b = normalize_preservation_choice(
                row.get("meaning_preservation_b")
            )

            # Modèle A
            if mod_a in stats:
                stats[mod_a]["eval_count"] += 1
                if simpler_a == SIMPLER_CHOICES[0]:
                    stats[mod_a]["simpler_oui_count"] += 1
                if pres_a == PRESERVATION_CHOICES[0]:
                    stats[mod_a]["sens_oui_count"] += 1
                if pref_opt == "A":
                    stats[mod_a]["pref_count"] += 1

            # Modèle B
            if mod_b in stats:
                stats[mod_b]["eval_count"] += 1
                if simpler_b == SIMPLER_CHOICES[0]:
                    stats[mod_b]["simpler_oui_count"] += 1
                if pres_b == PRESERVATION_CHOICES[0]:
                    stats[mod_b]["sens_oui_count"] += 1
                if pref_opt == "B":
                    stats[mod_b]["pref_count"] += 1

    records = []
    for m in models:
        cnt = stats[m]["eval_count"]
        simpler_pct = (stats[m]["simpler_oui_count"] / cnt * 100.0) if cnt > 0 else 0.0
        pres_pct = (stats[m]["sens_oui_count"] / cnt * 100.0) if cnt > 0 else 0.0
        pref_pct = (stats[m]["pref_count"] / cnt * 100.0) if cnt > 0 else 0.0
        records.append(
            {
                "Modèle": m,
                "Taux de Meilleure Simplification (%)": round(pref_pct, 1),
                "Taux de Contenu Plus Simple (%)": round(simpler_pct, 1),
                "Taux de Sens Conservé (%)": round(pres_pct, 1),
                "Taux de Préservation du Sens (%)": round(pres_pct, 1),
                "Taux de Préférence (%)": round(pref_pct, 1),
                "Total Évaluations": cnt,
                "Nb 'Oui' (Plus Simple)": stats[m]["simpler_oui_count"],
                "Nb 'Oui' (Sens)": stats[m]["sens_oui_count"],
                "Nb Préféré": stats[m]["pref_count"],
                "simpler_rate": round(simpler_pct, 1),
                "preference_rate": round(pref_pct, 1),
                "meaning_preservation_rate": round(pres_pct, 1),
                "simpler_yes_count": stats[m]["simpler_oui_count"],
                "meaning_preserved_count": stats[m]["sens_oui_count"],
                "preferred_count": stats[m]["pref_count"],
            }
        )

    return pd.DataFrame(records)


def generate_latex_human_study_macros(
    repo_root: Optional[Path] = None, force_partial: bool = False
) -> Dict[str, Any]:
    """Génère le dictionnaire et le bloc des 10 macros LaTeX pour Response_to_Reviewers.tex et Experiments.tex."""
    if repo_root is None:
        repo_root = find_repo_root()
    else:
        repo_root = Path(repo_root)

    iaa = compute_iaa_metrics(repo_root)
    df_perf = compute_model_performance(repo_root)

    henri_count = int(iaa.get("henri_total_annotated", 0))
    maxime_count = int(iaa.get("maxime_total_annotated", 0))
    overlap_count = int(iaa.get("annotated_shared_pairs", 0))

    is_complete = (henri_count >= 100 and maxime_count >= 100 and overlap_count >= 50)
    use_actual_values = is_complete or force_partial

    perf_map: Dict[str, Any] = {}
    if not df_perf.empty and "Modèle" in df_perf.columns:
        perf_map = df_perf.set_index("Modèle").to_dict(orient="index")

    mistral_data = perf_map.get("Mistral-7B-FT", {})
    gpt4o_data = perf_map.get("GPT-4o-FT", {})

    win_mistral = mistral_data.get("preference_rate", 0.0)
    win_gpt4o = gpt4o_data.get("preference_rate", 0.0)
    simpler_mistral = mistral_data.get("simpler_rate", 0.0)
    simpler_gpt4o = gpt4o_data.get("simpler_rate", 0.0)
    meaning_mistral = mistral_data.get("meaning_preservation_rate", 0.0)
    meaning_gpt4o = gpt4o_data.get("meaning_preservation_rate", 0.0)

    k_pref = iaa.get("cohen_kappa_pref", iaa.get("cohen_kappa"))
    k_simp = iaa.get("simpler_cohen_kappa")
    k_sem = iaa.get("meaning_cohen_kappa")
    alpha_krip = iaa.get("krippendorff_alpha")

    macros: Dict[str, str] = {}

    if overlap_count > 0 and use_actual_values:
        macros["HumanKappaPref"] = f"{k_pref:.2f}" if k_pref is not None else r"\textcolor{blue}{\textbf{[TBD: $\kappa_{\text{pref}}$]}}"
        macros["HumanKappaSimpler"] = f"{k_simp:.2f}" if k_simp is not None else r"\textcolor{blue}{\textbf{[TBD: $\kappa_{\text{simp}}$]}}"
        macros["HumanKappaMeaning"] = f"{k_sem:.2f}" if k_sem is not None else r"\textcolor{blue}{\textbf{[TBD: $\kappa_{\text{sem}}$]}}"
        macros["HumanAlphaKrip"] = f"{alpha_krip:.2f}" if alpha_krip is not None else r"\textcolor{blue}{\textbf{[TBD: $\alpha_{\text{Krip}}$]}}"
    else:
        macros["HumanKappaPref"] = r"\textcolor{blue}{\textbf{[TBD: $\kappa_{\text{pref}}$]}}"
        macros["HumanKappaSimpler"] = r"\textcolor{blue}{\textbf{[TBD: $\kappa_{\text{simp}}$]}}"
        macros["HumanKappaMeaning"] = r"\textcolor{blue}{\textbf{[TBD: $\kappa_{\text{sem}}$]}}"
        macros["HumanAlphaKrip"] = r"\textcolor{blue}{\textbf{[TBD: $\alpha_{\text{Krip}}$]}}"

    if use_actual_values and (henri_count > 0 or maxime_count > 0):
        macros["HumanWinMistral"] = f"{win_mistral:.1f}\\%"
        macros["HumanWinGPTFour"] = f"{win_gpt4o:.1f}\\%"
        macros["HumanSimplerMistral"] = f"{simpler_mistral:.1f}\\%"
        macros["HumanSimplerGPTFour"] = f"{simpler_gpt4o:.1f}\\%"
        macros["HumanMeaningMistral"] = f"{meaning_mistral:.1f}\\%"
        macros["HumanMeaningGPTFour"] = f"{meaning_gpt4o:.1f}\\%"
    else:
        macros["HumanWinMistral"] = r"\textcolor{blue}{\textbf{[TBD: \%]}}"
        macros["HumanWinGPTFour"] = r"\textcolor{blue}{\textbf{[TBD: \%]}}"
        macros["HumanSimplerMistral"] = r"\textcolor{blue}{\textbf{[TBD: \%]}}"
        macros["HumanSimplerGPTFour"] = r"\textcolor{blue}{\textbf{[TBD: \%]}}"
        macros["HumanMeaningMistral"] = r"\textcolor{blue}{\textbf{[TBD: \%]}}"
        macros["HumanMeaningGPTFour"] = r"\textcolor{blue}{\textbf{[TBD: \%]}}"

    order = [
        "HumanKappaPref",
        "HumanKappaSimpler",
        "HumanKappaMeaning",
        "HumanAlphaKrip",
        "HumanWinMistral",
        "HumanWinGPTFour",
        "HumanSimplerMistral",
        "HumanSimplerGPTFour",
        "HumanMeaningMistral",
        "HumanMeaningGPTFour",
    ]

    latex_lines = [
        "% Macros for final human evaluation numbers (Task T14, N=150 stratified pairs)"
    ]
    for key in order:
        latex_lines.append(f"\\newcommand{{\\{key}}}{{{macros[key]}}}")

    latex_commands_block = "\n".join(latex_lines)

    return {
        "is_complete": is_complete,
        "henri_count": henri_count,
        "maxime_count": maxime_count,
        "overlap_count": overlap_count,
        "macros_dict": macros,
        "latex_commands_block": latex_commands_block,
    }
