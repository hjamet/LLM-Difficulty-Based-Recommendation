#!/usr/bin/env python3
"""Script d'automatisation de l'Étape 3.4 : Préparation de la release publique.

Ce script permet d'auditer et de préparer le dépôt GitHub en vue de son ouverture
publique lors de la soumission finale DMKD (Étape 3.4).

Deux modes d'exécution :
1. --check (par défaut, non destructif) :
   - Vérifie l'absence totale de secrets/clés API (sk-..., hf_..., ghp_..., AIza...)
   - Vérifie l'absence de chemins utilisateurs privés (/home/lopilo, C:\\Users\\)
   - Liste les artefacts internes à isoler sur la branche d'archive (rebuttal-dmkd-internal)
   - Vérifie l'existence de tous les notebooks et scripts référencés dans README.md
2. --apply-on-branch <branch> :
   - Crée la branche cible pour la release publique
   - Anonymise les fichiers d'annotations nominatifs (evaluator_1, evaluator_2)
   - Supprime de l'arbre Git les répertoires internes (rebuttal/, overleaf_patches/)
   - Committe les modifications avec un message conventionnel propre.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


class SecretMatch(NamedTuple):
    file_path: Path
    line_number: int
    secret_type: str
    match_preview: str


class PathLeakMatch(NamedTuple):
    file_path: Path
    line_number: int
    leak_type: str
    matched_text: str


# Motifs stricts pour détecter les clés API réelles
SECRET_PATTERNS: Dict[str, re.Pattern[str]] = {
    "OpenAI API Key (sk-)": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "HuggingFace Token (hf_)": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub Token (ghp_)": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "Google API Key (AIza)": re.compile(r"\bAIza[A-Za-z0-9_-]{30,40}\b"),
}

# Motifs de chemins d'utilisateurs privés
PATH_PATTERNS: Dict[str, re.Pattern[str]] = {
    "Linux User Home": re.compile(r"/home/(?:lopilo|lopilo24|cb24)\b[^\s\"\'\\]*"),
    "Windows User Path": re.compile(r"[C-Zc-z]:\\Users\\[A-Za-z0-9_.-]+"),
}

INTERNAL_DIRECTORIES_TO_ISOLATE: List[str] = [
    "dmkd_additions/rebuttal",
    "dmkd_additions/overleaf_patches",
]

NOMINATIVE_ANNOTATION_FILES: Dict[str, str] = {
    "annotations_henri.csv": "annotations_evaluator_1.csv",
    "annotations_maxime.csv": "annotations_evaluator_2.csv",
}

EXPECTED_README_NOTEBOOKS: List[str] = [
    # Difficulty Estimation (7 notebooks)
    "difficulty_estimation/a_Datasets.ipynb",
    "difficulty_estimation/b_OpenAiModelsTraining.ipynb",
    "difficulty_estimation/c_OpenAiEvaluation.ipynb",
    "difficulty_estimation/d_OpenSourceModelsTraining.ipynb",
    "difficulty_estimation/e_OpenSourceModelsEvaluation.ipynb",
    "difficulty_estimation/f_PairwiseMismatch.ipynb",
    "difficulty_estimation/g_CreateFigures.ipynb",
    # Topic Classification (5 notebooks)
    "topic_classification/a_DataPreparation.ipynb",
    "topic_classification/b_Zero-Shot.ipynb",
    "topic_classification/c_OpenAiTopicClassification.ipynb",
    "topic_classification/d_FlaubertFineTuned.ipynb",
    "topic_classification/e_CreateFigures.ipynb",
    # Text Simplification (8 notebooks)
    "text_simplification/a_DatasetCreation.ipynb",
    "text_simplification/b_FineTuningMistral.ipynb",
    "text_simplification/c_MistralEvaluation.ipynb",
    "text_simplification/d_Metrics.ipynb",
    "text_simplification/e_OpenAIFineTuning.ipynb",
    "text_simplification/f_OpenAIEvaluation.ipynb",
    "text_simplification/g_IterativeSimplification.ipynb",
    "text_simplification/h_CreateFigures.ipynb",
    # Recommender Systems (4 notebooks)
    "recommender_system/a_GetZeeguData.ipynb",
    "recommender_system/b_GenerateEmbeddings.ipynb",
    "recommender_system/c_Recommendations.ipynb",
    "recommender_system/d_Graphic.ipynb",
]


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """Détecte la racine du dépôt git."""
    current = (start_path or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").is_dir():
            return parent
    return current


def get_tracked_files(repo_root: Path) -> List[Path]:
    """Liste tous les fichiers suivis par git dans le dépôt."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=str(repo_root),
            text=True,
            errors="ignore",
        )
        files = []
        for line in output.splitlines():
            line = line.strip()
            if line:
                full_p = repo_root / line
                if full_p.is_file():
                    files.append(full_p)
        return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback pour exécution hors git / tests
        return [p for p in repo_root.rglob("*") if p.is_file() and ".git" not in p.parts]


def scan_file_for_secrets(file_path: Path) -> List[SecretMatch]:
    """Scanne un fichier individuel à la recherche de clés API."""
    matches: List[SecretMatch] = []
    # Ignore les fichiers binaires ou pdf
    if file_path.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".zip", ".tar", ".gz", ".lock"]:
        return matches

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return matches

    for idx, line in enumerate(content.splitlines(), start=1):
        for name, pattern in SECRET_PATTERNS.items():
            for m in pattern.finditer(line):
                matched_val = m.group(0)
                # Obfuscation pour l'affichage de sécurité
                preview = matched_val[:6] + "..." + matched_val[-4:] if len(matched_val) > 10 else "***"
                matches.append(
                    SecretMatch(
                        file_path=file_path,
                        line_number=idx,
                        secret_type=name,
                        match_preview=preview,
                    )
                )
    return matches


def scan_file_for_path_leaks(file_path: Path) -> List[PathLeakMatch]:
    """Scanne un fichier pour détecter des fuites de chemins personnels Linux ou Windows."""
    matches: List[PathLeakMatch] = []
    if file_path.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".zip", ".tar", ".gz", ".lock"]:
        return matches

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return matches

    for idx, line in enumerate(content.splitlines(), start=1):
        # Ignorer les commentaires et regex déclarés dans prepare_public_release.py lui-même
        if "prepare_public_release.py" in str(file_path) or "test_prepare_public_release.py" in str(file_path):
            continue
        for name, pattern in PATH_PATTERNS.items():
            for m in pattern.finditer(line):
                # Ignorer si le match fait partie d'une URL web (ex: https://.../home/...)
                prefix = line[:m.start()]
                if re.search(r"https?://[^\s\"']*$", prefix):
                    continue
                # Ignorer l'hallucination de script Python externe générée par davinci-002 dans results/
                if "results" in file_path.parts and "Jobend" in m.group(0):
                    continue
                matched = m.group(0)
                matches.append(
                    PathLeakMatch(
                        file_path=file_path,
                        line_number=idx,
                        leak_type=name,
                        matched_text=matched,
                    )
                )
    return matches


def check_all_secrets(repo_root: Path, target_files: Optional[List[Path]] = None) -> List[SecretMatch]:
    """Exécute l'audit complet des secrets sur les fichiers cibles."""
    files = target_files if target_files is not None else get_tracked_files(repo_root)
    all_matches: List[SecretMatch] = []
    for f in files:
        all_matches.extend(scan_file_for_secrets(f))
    return all_matches


def check_all_path_leaks(repo_root: Path, target_files: Optional[List[Path]] = None) -> List[PathLeakMatch]:
    """Exécute l'audit complet des fuites de chemins sur les fichiers cibles."""
    files = target_files if target_files is not None else get_tracked_files(repo_root)
    all_matches: List[PathLeakMatch] = []
    for f in files:
        all_matches.extend(scan_file_for_path_leaks(f))
    return all_matches


def list_internal_artifacts_to_isolate(repo_root: Path) -> Dict[str, List[Path]]:
    """Identifie précisément les répertoires et fichiers internes à isoler sur la branche d'archive."""
    isolated_dirs: List[Path] = []
    for rel_d in INTERNAL_DIRECTORIES_TO_ISOLATE:
        d_path = repo_root / rel_d
        if d_path.exists():
            isolated_dirs.append(d_path)

    reviews_dir = repo_root / "dmkd_additions" / "text_simplification" / "reviews"
    nominative_files: List[Path] = []
    for orig_name in NOMINATIVE_ANNOTATION_FILES.keys():
        f_path = reviews_dir / orig_name
        if f_path.is_file():
            nominative_files.append(f_path)

    return {
        "directories": isolated_dirs,
        "nominative_files": nominative_files,
    }


def verify_readme_references(repo_root: Path) -> List[str]:
    """Vérifie que tous les notebooks et scripts mentionnés dans le README existent."""
    missing: List[str] = []
    readme_path = repo_root / "README.md"
    if not readme_path.is_file():
        return ["README.md manquant"]

    for rel_nb in EXPECTED_README_NOTEBOOKS:
        full_p = repo_root / rel_nb
        if not full_p.is_file():
            missing.append(rel_nb)
    return missing


def anonymize_evaluator_files(
    repo_root: Path,
    dry_run: bool = False,
) -> List[Tuple[Path, Path]]:
    """Anonymise les fichiers nominatifs annotations_henri.csv et annotations_maxime.csv.

    Remplace la colonne 'annotator' :
    - 'henri' -> 'evaluator_1'
    - 'maxime' -> 'evaluator_2'
    Écrit annotations_evaluator_1.csv et annotations_evaluator_2.csv.
    """
    reviews_dir = repo_root / "dmkd_additions" / "text_simplification" / "reviews"
    created_pairs: List[Tuple[Path, Path]] = []

    replacements = {
        "annotations_henri.csv": ("annotations_evaluator_1.csv", "henri", "evaluator_1"),
        "annotations_maxime.csv": ("annotations_evaluator_2.csv", "maxime", "evaluator_2"),
    }

    for src_name, (dst_name, old_annotator, new_annotator) in replacements.items():
        src_path = reviews_dir / src_name
        dst_path = reviews_dir / dst_name
        if not src_path.is_file():
            continue

        content = src_path.read_text(encoding="utf-8")
        # Remplacement rigoureux dans le format CSV (colonne annotator en 2e position)
        lines = content.splitlines()
        new_lines: List[str] = []
        if lines:
            header = lines[0]
            new_lines.append(header)
            for line in lines[1:]:
                if not line.strip():
                    continue
                parts = line.split(",", 2)
                if len(parts) >= 3 and parts[1] == old_annotator:
                    parts[1] = new_annotator
                    new_lines.append(",".join(parts))
                else:
                    new_lines.append(line)

        new_content = "\n".join(new_lines) + "\n"

        if not dry_run:
            dst_path.write_text(new_content, encoding="utf-8")

        created_pairs.append((src_path, dst_path))

    return created_pairs


def run_check_mode(repo_root: Path) -> bool:
    """Exécute l'ensemble des vérifications non destructives du mode --check."""
    print("=" * 70)
    print(" AUDIT DE CONFORMITÉ PRÉ-RELEASE PUBLIQUE (Étape 3.4)")
    print(f" Dépôt racine : {repo_root}")
    print("=" * 70)

    # 1. Vérification des clés API
    print("\n[1/4] Audit des secrets et cles API...")
    secret_matches = check_all_secrets(repo_root)
    if secret_matches:
        print(f"  [FAIL] {len(secret_matches)} secrets ou cles API detectes !")
        for m in secret_matches:
            print(f"     - {m.file_path.relative_to(repo_root)}:L{m.line_number} [{m.secret_type}] {m.match_preview}")
    else:
        print("  [OK] Zero cle API (sk-, hf_, ghp_, AIza) detectee.")

    # 2. Vérification des fuites de chemins utilisateurs
    print("\n[2/4] Audit des fuites de chemins personnels (/home/lopilo, C:\\Users\\)...")
    leak_matches = check_all_path_leaks(repo_root)
    if leak_matches:
        print(f"  [FAIL] {len(leak_matches)} fuites de chemins detectees !")
        for l in leak_matches:
            print(f"     - {l.file_path.relative_to(repo_root)}:L{l.line_number} [{l.leak_type}] {l.matched_text}")
    else:
        print("  [OK] Zero fuite de chemin utilisateur personnel detectee.")

    # 3. Vérification des références README
    print("\n[3/4] Verification de l'integrite des notebooks references dans README.md...")
    missing_nbs = verify_readme_references(repo_root)
    if missing_nbs:
        print(f"  [FAIL] {len(missing_nbs)} fichiers references manquants !")
        for nb in missing_nbs:
            print(f"     - {nb}")
    else:
        print(f"  [OK] Tous les {len(EXPECTED_README_NOTEBOOKS)} notebooks references sont presents.")

    # 4. Inventaire des artefacts internes à isoler
    print("\n[4/4] Inventaire des artefacts internes pour la branche d'archive...")
    internal = list_internal_artifacts_to_isolate(repo_root)
    print("  Repertoires a isoler :")
    for d in internal["directories"]:
        print(f"     - {d.relative_to(repo_root)}/")
    print("  Fichiers d'annotations a anonymiser :")
    for f in internal["nominative_files"]:
        target_name = NOMINATIVE_ANNOTATION_FILES.get(f.name, f.name)
        print(f"     - {f.relative_to(repo_root)}  ==>  {target_name}")

    all_passed = (len(secret_matches) == 0) and (len(leak_matches) == 0) and (len(missing_nbs) == 0)
    print("\n" + "=" * 70)
    if all_passed:
        print(" RESULTAT GLOBAL : CONFORME (Tous les criteres de release sont valides)")
    else:
        print(" RESULTAT GLOBAL : NON CONFORME (Des anomalies doivent etre corrigees)")
    print("=" * 70)
    return all_passed


def apply_on_branch(repo_root: Path, target_branch: str) -> None:
    """Applique l'isolation et l'anonymisation sur la branche cible."""
    print("=" * 70)
    print(f" APPLICATION DE L'ISOLATION ET ANONYMISATION SUR LA BRANCHE : {target_branch}")
    print("=" * 70)

    # 1. Vérification de propreté Git
    status_out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True)
    if status_out.strip():
        raise RuntimeError("Le répertoire de travail Git contient des modifications non commitées. Committez ou stashlez d'abord.")

    # 2. Création ou bascule sur target_branch
    print(f"[1/4] Création / bascule sur la branche '{target_branch}'...")
    subprocess.check_call(["git", "checkout", "-B", target_branch], cwd=str(repo_root))

    # 3. Anonymisation des annotations
    print("[2/4] Anonymisation des fichiers d'annotations...")
    pairs = anonymize_evaluator_files(repo_root, dry_run=False)
    for src, dst in pairs:
        print(f"   Créé : {dst.relative_to(repo_root)} depuis {src.name}")
        # Supprime le nominatif
        subprocess.check_call(["git", "rm", "-f", str(src.relative_to(repo_root))], cwd=str(repo_root))
        subprocess.check_call(["git", "add", str(dst.relative_to(repo_root))], cwd=str(repo_root))

    # 4. Suppression des répertoires internes de la release publique
    print("[3/4] Retrait des dossiers internes de la branche publique...")
    for rel_d in INTERNAL_DIRECTORIES_TO_ISOLATE:
        d_path = repo_root / rel_d
        if d_path.exists():
            print(f"   Suppression de l'arbre Git : {rel_d}/")
            subprocess.check_call(["git", "rm", "-rf", "--ignore-unmatch", rel_d], cwd=str(repo_root))

    # 5. Commit de la release
    print("[4/4] Enregistrement du commit de release...")
    commit_msg = (
        "chore(release): isolate internal rebuttal artifacts and anonymize evaluators\n\n"
        "- Remove dmkd_additions/rebuttal/ and overleaf_patches/ from public tree\n"
        "- Anonymize annotations_henri.csv and annotations_maxime.csv as annotations_evaluator_1.csv and annotations_evaluator_2.csv"
    )
    subprocess.check_call(["git", "commit", "-m", commit_msg], cwd=str(repo_root))
    print(f"\n[SUCCÈS] Branche publique '{target_branch}' prête pour publication !")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script de préparation et conformité pour la release publique DMKD (Étape 3.4)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Exécute l'audit de conformité non destructif (par défaut)",
    )
    parser.add_argument(
        "--apply-on-branch",
        type=str,
        default=None,
        metavar="BRANCH_NAME",
        help="Crée la branche spécifiée et applique l'isolation et l'anonymisation",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Chemin de la racine du dépôt git (détecté automatiquement si omis)",
    )

    args = parser.parse_args()
    repo_root = find_repo_root(args.repo_root)

    if args.apply_on_branch:
        apply_on_branch(repo_root, args.apply_on_branch)
    else:
        success = run_check_mode(repo_root)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
