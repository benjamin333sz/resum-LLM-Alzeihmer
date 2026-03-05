from typing import Any
from pathlib import Path


def load_prompt_file(file_path: str) -> str:
    """Lit le contenu d’un fichier de prompt et le retourne."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return path.read_text(encoding="utf-8").strip()


def prompt_needs_update(remote_text: str, local_text: str) -> bool:
    """Vérifie si le contenu local diffère du contenu distant."""
    return (remote_text or "").strip() != local_text.strip()
