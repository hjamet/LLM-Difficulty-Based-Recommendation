"""Tests unitaires pour dmkd_additions/scripts/prepare_public_release.py.

Vérifie :
1. La détection et le filtrage des clés API (vrais positifs vs faux positifs)
2. La détection des fuites de chemins personnels (/home/lopilo, C:\\Users\\)
3. La validation des références de notebooks dans README.md
4. L'identification exacte des artefacts internes à isoler
5. L'anonymisation rigoureuse des annotateurs
6. L'exécution bout-en-bout du mode check sur le dépôt actuel.
"""

import sys
from pathlib import Path
import tempfile
import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dmkd_additions.scripts.prepare_public_release import (
    check_all_path_leaks,
    check_all_secrets,
    find_repo_root,
    list_internal_artifacts_to_isolate,
    run_check_mode,
    scan_file_for_path_leaks,
    scan_file_for_secrets,
    verify_readme_references,
)


@pytest.fixture
def repo_root() -> Path:
    return find_repo_root()


def test_secret_detection_true_and_false_positives(tmp_path: Path):
    """Vérifie que les clés API sont détectées et les faux positifs ignorés."""
    # Faux positifs courants (mots contenant sk-)
    clean_file = tmp_path / "clean.py"
    clean_file.write_text(
        "abstract = 'This task-specific model uses masked language modeling.'\n"
        "url = 'https://example.com/logomask-2014.png'\n"
        "short_sk = 'sk-123'\n",
        encoding="utf-8",
    )
    matches_clean = scan_file_for_secrets(clean_file)
    assert len(matches_clean) == 0, f"Faux positifs détectés : {matches_clean}"

    # Vrais secrets (construits par concaténation dynamique pour éviter l'auto-détection statique sur ce fichier)
    fake_sk = "sk-" + "abcdef1234567890abcdef1234567890"
    fake_hf = "hf_" + "abcdef1234567890abcdef1234"
    fake_ghp = "ghp_" + "abcdef1234567890abcdef1234567890"
    fake_aiza = "AIza" + "SyA1234567890123456789012345678901"
    dirty_file = tmp_path / "dirty.py"
    dirty_file.write_text(
        f"openai_key = '{fake_sk}'\n"
        f"hf_token = '{fake_hf}'\n"
        f"gh_token = '{fake_ghp}'\n"
        f"google_key = '{fake_aiza}'\n",
        encoding="utf-8",
    )
    matches_dirty = scan_file_for_secrets(dirty_file)
    assert len(matches_dirty) == 4
    types = {m.secret_type for m in matches_dirty}
    assert "OpenAI API Key (sk-)" in types
    assert "HuggingFace Token (hf_)" in types
    assert "GitHub Token (ghp_)" in types
    assert "Google API Key (AIza)" in types


def test_path_leak_detection(tmp_path: Path):
    """Vérifie que les chemins utilisateurs Linux et Windows sont détectés."""
    dirty_path_file = tmp_path / "leaks.txt"
    dirty_path_file.write_text(
        "Error in /home/lopilo/code/project/main.py\n"
        "Cached in /home/lopilo24/.cache/huggingface\n"
        "Windows path C:\\Users\\Henri\\Documents\\code\n"
        "Legit URL https://www.lutinbazar.fr/home/cb24-2-2-2.html\n",
        encoding="utf-8",
    )
    leaks = scan_file_for_path_leaks(dirty_path_file)
    # Devrait détecter /home/lopilo, /home/lopilo24 et C:\Users\Henri, mais PAS l'URL lutinbazar
    assert len(leaks) == 3
    leak_texts = [l.matched_text for l in leaks]
    assert any("/home/lopilo" in t for t in leak_texts)
    assert any("/home/lopilo24" in t for t in leak_texts)
    assert any("C:\\Users\\Henri" in t for t in leak_texts)
    assert not any("lutinbazar" in t for t in leak_texts)


def test_readme_references_on_real_repo(repo_root: Path):
    """Vérifie que tous les 24 notebooks décrits dans README.md existent sur le dépôt."""
    missing = verify_readme_references(repo_root)
    assert missing == [], f"Notebooks manquants référencés par le README : {missing}"


def test_internal_artifacts_inventory(repo_root: Path):
    """Vérifie l'inventaire des répertoires et fichiers à isoler sur rebuttal-dmkd-internal."""
    artifacts = list_internal_artifacts_to_isolate(repo_root)
    dir_names = [d.name for d in artifacts["directories"]]
    assert "rebuttal" in dir_names
    assert "overleaf_patches" in dir_names

    file_names = [f.name for f in artifacts["nominative_files"]]
    assert "annotations_henri.csv" in file_names
    assert "annotations_maxime.csv" in file_names


def test_anonymize_evaluators_content(tmp_path: Path):
    """Vérifie que l'anonymisation transforme henri -> evaluator_1 et maxime -> evaluator_2."""
    from dmkd_additions.scripts.prepare_public_release import anonymize_evaluator_files

    # Crée un faux repo structure
    rev_dir = tmp_path / "dmkd_additions" / "text_simplification" / "reviews"
    rev_dir.mkdir(parents=True)

    h_file = rev_dir / "annotations_henri.csv"
    h_file.write_text(
        "pair_id,annotator,timestamp,choice\n"
        "P001,henri,2026-10-20T14:00:00,A\n",
        encoding="utf-8",
    )

    m_file = rev_dir / "annotations_maxime.csv"
    m_file.write_text(
        "pair_id,annotator,timestamp,choice\n"
        "P002,maxime,2026-10-23T14:00:00,B\n",
        encoding="utf-8",
    )

    created = anonymize_evaluator_files(tmp_path, dry_run=False)
    assert len(created) == 2

    e1 = rev_dir / "annotations_evaluator_1.csv"
    e2 = rev_dir / "annotations_evaluator_2.csv"
    assert e1.is_file()
    assert e2.is_file()

    c1 = e1.read_text(encoding="utf-8")
    assert "evaluator_1" in c1
    assert "henri" not in c1

    c2 = e2.read_text(encoding="utf-8")
    assert "evaluator_2" in c2
    assert "maxime" not in c2


def test_run_check_mode_on_active_repository(repo_root: Path):
    """Vérifie que le dépôt actif (après nettoyage des notebooks) passe 100% du mode check."""
    passed = run_check_mode(repo_root)
    assert passed is True, "Le mode --check sur le dépôt actif doit être 100% conforme."
