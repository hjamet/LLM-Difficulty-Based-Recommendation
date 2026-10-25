#!/usr/bin/env python3
r"""Script CLI pour previsualiser et injecter automatiquement les resultats de l'etude humaine

dans Response_to_Reviewers.tex et dmkd_additions/overleaf_patches/tex/Experiments.tex.
Supporte --preview-partial et --apply (avec --force si etude partielle).
"""

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Dict

# Securiser stdout pour Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Assurer la presence de repo_root dans sys.path
_current_dir = Path(__file__).resolve().parent
_repo_root = _current_dir.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dmkd_additions.text_simplification.reviews.annotation_engine import (
    find_repo_root,
    generate_latex_human_study_macros,
)


def update_target_file(file_path: Path, macros_dict: Dict[str, str]) -> int:
    r"""Met a jour les definitions \newcommand{\Human...} ou \providecommand{\Human...} dans file_path."""
    if not file_path.is_file():
        print(f"[WARN] Fichier cible introuvable : {file_path}")
        return 0

    content = file_path.read_text(encoding="utf-8")
    updated_count = 0

    for key, val in macros_dict.items():
        pattern = re.compile(
            r"(\\(?:newcommand|providecommand)\{\\" + re.escape(key) + r"\}\{)(?:.*?)(\})"
        )
        if pattern.search(content):
            def _repl(m: re.Match) -> str:
                return f"{m.group(1)}{val}{m.group(2)}"

            new_content = pattern.sub(_repl, content)
            if new_content != content:
                content = new_content
                updated_count += 1

    if updated_count > 0:
        file_path.write_text(content, encoding="utf-8")
        print(f"[OK] {file_path.name} : {updated_count} macros mises a jour.")
    else:
        print(f"[INFO] {file_path.name} : aucune macro a modifier ou deja a jour.")

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Mise a jour automatique des macros LaTeX de l'etude d'annotation humaine (T14)."
    )
    parser.add_argument(
        "--preview-partial",
        action="store_true",
        help="Affiche les macros calculees sur les donnees partielles existantes.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Injecte les macros dans Response_to_Reviewers.tex et Experiments.tex.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Calculer et injecter les métriques en excluant les 11 paires atypiques (NaN regex, prompt leak, A==B).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force l'injection des valeurs partielles meme si l'etude n'est pas complete.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Chemin vers la racine du depot git (detecte automatiquement par defaut).",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else find_repo_root()
    force_partial = args.preview_partial or args.force

    res_raw = generate_latex_human_study_macros(repo_root, force_partial=force_partial, exclude_anomalies=False)
    res_clean = generate_latex_human_study_macros(repo_root, force_partial=force_partial, exclude_anomalies=True)

    res = res_clean if args.clean else res_raw

    print("=" * 75)
    print("ETUDE D'ANNOTATION HUMAINE DMKD (T14) - BILAN COMPARATIF DES DONNEES")
    print("=" * 75)
    print("1. POOL COMPLET (N=150, n_shared=50) :")
    print(f"  * Progression Henri   : {res_raw['henri_count']} / 100")
    print(f"  * Progression Maxime  : {res_raw['maxime_count']} / 100")
    print(f"  * Paires Communes     : {res_raw['overlap_count']} / 50")
    print(f"  * Statut Completude   : {'COMPLETE (150/150 paires)' if res_raw['is_complete'] else 'EN COURS (partielle)'}")
    print()
    print("2. POOL NETTOYE HORS ANOMALIES (N_clean=139, n_shared_clean=48) :")
    print("  * Paires NaN regex (?/!)    (5) : P014, P032, P042, P075, P091")
    print("  * Prompt leak [/INST]       (1) : P074")
    print("  * Paires identiques A==B    (5) : P030, P049, P056, P108, P143")
    print(f"  * Paires communes nettoyees : {res_clean['overlap_count']} / 48")
    print(f"  * Mode actif                : {'POOL NETTOYE (--clean)' if args.clean else 'POOL COMPLET (standard)'}")
    print("=" * 75)

    if args.preview_partial or (not args.apply):
        print(f"\nBLOC DE MACROS LATEX GENERE ({'POOL NETTOYE' if args.clean else 'POOL COMPLET'}) :")
        print(res["latex_commands_block"])
        print()

    if args.apply:
        if not res["is_complete"] and not args.force:
            print("[ERROR] L'etude n'est pas encore complete.")
            print("   Pour injecter les chiffres partiels actuels, utilisez l'option --force :")
            print("   python update_human_study_latex.py --apply --force" + (" --clean" if args.clean else ""))
            sys.exit(1)

        print(f"[APPLY] Application des macros ({'POOL NETTOYE' if args.clean else 'POOL COMPLET'}) dans les fichiers sources...")
        targets = [
            repo_root / "dmkd_additions" / "rebuttal" / "Response_to_Reviewers.tex",
            repo_root / "dmkd_additions" / "overleaf_patches" / "tex" / "Experiments.tex",
        ]

        total_mods = 0
        for tgt in targets:
            total_mods += update_target_file(tgt, res["macros_dict"])

        print(f"\n[DONE] Termine avec succes : {total_mods} modifications appliquees.")


if __name__ == "__main__":
    main()
