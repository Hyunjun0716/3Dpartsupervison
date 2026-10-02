"""Repository-owned defaults; explicit CLI paths remain caller-relative."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(relative_path):
    return str(PROJECT_ROOT / relative_path)
