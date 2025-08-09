from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent.parent / 'config' / 'hipaa_config.yml'

# Define sensitive environment variables that should be loaded securely
SENSITIVE_ENV_VARS = {
    "epic_api_key": "EPIC_API_KEY",  # pragma: allowlist secret
    "azure_key_vault_url": "AZURE_KEY_VAULT_URL",
    "encryption_key": "ENCRYPTION_KEY",
    "database_url": "DATABASE_URL",
    "redis_url": "REDIS_URL",
    "openai_api_key": "OPENAI_API_KEY",  # pragma: allowlist secret
    "aws_access_key_id": "AWS_ACCESS_KEY_ID",
    "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",  # pragma: allowlist secret
    "jwt_secret_key": "JWT_SECRET_KEY"  # pragma: allowlist secret
}


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


def get_secret_config() -> Dict[str, Any]:
    """Load sensitive configuration from environment variables.
    
    Returns a dictionary with sensitive configuration values loaded from 
    environment variables. Values that are not set will be None.
    """
    config = {}

    for config_key, env_var in SENSITIVE_ENV_VARS.items():
        value = os.environ.get(env_var)
        config[config_key] = value

        if value:
            logger.info(f"Loaded {config_key} from environment variable {env_var}")
        else:
            logger.debug(f"Environment variable {env_var} not set for {config_key}")

    return config


def validate_secret_config(config: Dict[str, Any], required_for_production: bool = False) -> List[str]:
    """Validate that required secrets are present.
    
    Args:
        config: Configuration dictionary to validate
        required_for_production: If True, enforce stricter validation for production
        
    Returns:
        List of validation error messages
    """
    errors = []

    # Define which secrets are required in production
    if required_for_production:
        required_secrets = ["encryption_key"]
        recommended_secrets = ["epic_api_key", "database_url"]

        for secret in required_secrets:
            if not config.get(secret):
                errors.append(f"Required secret '{secret}' is not configured")

        for secret in recommended_secrets:
            if not config.get(secret):
                errors.append(f"Recommended secret '{secret}' is not configured for production")

    # Validate URL format for URL-based secrets
    url_secrets = ["azure_key_vault_url", "database_url", "redis_url"]
    for secret in url_secrets:
        value = config.get(secret)
        if value:
            try:
                parsed = urlparse(value)
                if not parsed.scheme or not parsed.netloc:
                    errors.append(f"Invalid URL format for '{secret}': {value}")
            except Exception as e:
                errors.append(f"Error parsing URL for '{secret}': {str(e)}")

    return errors


def mask_sensitive_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive values for safe logging.
    
    Args:
        config: Configuration dictionary with potentially sensitive values
        
    Returns:
        Dictionary with sensitive values masked
    """
    masked = config.copy()

    # Define patterns that indicate sensitive data - be specific to avoid false positives
    sensitive_patterns = ['api_key', '_key', 'token', 'password']
    # Check for secret but not if preceded by "non" or "not"

    for key, value in masked.items():
        if value and isinstance(value, str):
            # Only mask if the key name indicates it's sensitive
            is_sensitive_key = any(pattern in key.lower() for pattern in sensitive_patterns)

            # Special handling for "secret" to avoid matching "non_secret"
            if 'secret' in key.lower() and not any(prefix in key.lower() for prefix in ['non_', 'not_']):
                is_sensitive_key = True

            if is_sensitive_key:
                if len(value) > 10:
                    masked[key] = value[:6] + "***"
                else:
                    masked[key] = "***"

            # Mask URLs that contain passwords
            elif 'url' in key.lower():
                try:
                    parsed = urlparse(value)
                    if parsed.password:
                        # Replace password in URL
                        netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
                        masked_url = value.replace(parsed.netloc, netloc)
                        masked[key] = masked_url
                    elif '://' not in value:
                        # Not a valid URL format, mask it for safety
                        logger.warning(f"Invalid URL format detected for key '{key}': {value}")
                        masked[key] = "***"
                except Exception as e:
                    # If parsing fails, log the issue and mask the whole thing
                    logger.warning(f"Failed to parse URL for masking key '{key}': {e}")
                    masked[key] = "***"

    return masked


CONFIG = load_config()

__all__ = ["CONFIG", "load_config", "get_secret_config", "validate_secret_config", "mask_sensitive_config"]
