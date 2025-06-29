from __future__ import annotations

from pathlib import Path
import os
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / 'config' / 'hipaa_config.yml'


def load_config(path: str | Path | None = None) -> dict:
    """Load configuration from environment variable or path.

    Priority is given to the ``HIPAA_CONFIG_YAML`` environment variable. If
    unset, ``HIPAA_CONFIG_PATH`` or the provided ``path`` will be used, falling
    back to the packaged default file. Returns an empty configuration if no
    source is available.
    """

    env_yaml = os.environ.get("HIPAA_CONFIG_YAML")
    if env_yaml:
        try:
            return yaml.safe_load(env_yaml) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - invalid YAML rarely
            raise ValueError("Invalid YAML in HIPAA_CONFIG_YAML") from exc

    path = Path(os.environ.get("HIPAA_CONFIG_PATH", path or DEFAULT_PATH))
    if path.exists():
        with path.open("r") as fh:
            return yaml.safe_load(fh) or {}
    return {}


CONFIG = load_config()

__all__ = ["CONFIG", "load_config"]
