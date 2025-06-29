from __future__ import annotations

from pathlib import Path
import os
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / 'config' / 'hipaa_config.yml'


def load_config(path: str | Path | None = None) -> dict:
    """Load configuration from ``path`` or default path."""
    path = Path(os.environ.get('HIPAA_CONFIG_PATH', path or DEFAULT_PATH))
    if path.exists():
        with path.open('r') as fh:
            return yaml.safe_load(fh) or {}
    return {}


CONFIG = load_config()

__all__ = ["CONFIG", "load_config"]
