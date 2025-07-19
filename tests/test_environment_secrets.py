import os
import pytest
from pathlib import Path
from unittest.mock import patch


def test_config_loads_from_environment_variables():
    """Test that sensitive configuration is loaded from environment variables."""
    from hipaa_compliance_summarizer.config import load_config
    
    # Test environment variables for sensitive data
    env_vars = {
        "EPIC_API_KEY": "test_epic_key_123",  # pragma: allowlist secret
        "AZURE_KEY_VAULT_URL": "https://test-vault.vault.azure.net/",
        "ENCRYPTION_KEY": "test_encryption_key_base64",
        "DATABASE_URL": "postgresql://user:pass@localhost/hipaa_db",  # pragma: allowlist secret
        "REDIS_URL": "redis://localhost:6379/0"
    }
    
    with patch.dict(os.environ, env_vars):
        # Import after patching environment
        from hipaa_compliance_summarizer.config import get_secret_config
        
        config = get_secret_config()
        
        # Verify sensitive values are loaded from environment
        assert config.get("epic_api_key") == "test_epic_key_123"
        assert config.get("azure_key_vault_url") == "https://test-vault.vault.azure.net/"
        assert config.get("encryption_key") == "test_encryption_key_base64"
        assert config.get("database_url") == "postgresql://user:pass@localhost/hipaa_db"  # pragma: allowlist secret
        assert config.get("redis_url") == "redis://localhost:6379/0"


def test_config_has_secure_defaults():
    """Test that configuration has secure defaults when environment vars are missing."""
    from hipaa_compliance_summarizer.config import get_secret_config
    
    # Clear any existing environment variables
    env_vars_to_clear = [
        "EPIC_API_KEY", "AZURE_KEY_VAULT_URL", "ENCRYPTION_KEY", 
        "DATABASE_URL", "REDIS_URL"
    ]
    
    with patch.dict(os.environ, {}, clear=True):
        config = get_secret_config()
        
        # Should have secure defaults or None for missing secrets
        assert config.get("epic_api_key") is None
        assert config.get("azure_key_vault_url") is None
        assert config.get("encryption_key") is None
        assert config.get("database_url") is None
        assert config.get("redis_url") is None


def test_config_validation_for_required_secrets():
    """Test that configuration validation catches missing required secrets."""
    from hipaa_compliance_summarizer.config import validate_secret_config
    
    # Test with missing required secrets
    incomplete_config = {
        "epic_api_key": None,
        "encryption_key": None
    }
    
    errors = validate_secret_config(incomplete_config, required_for_production=True)
    
    # Should identify missing required secrets
    assert len(errors) > 0
    assert any("epic_api_key" in error for error in errors)
    assert any("encryption_key" in error for error in errors)


def test_config_masking_in_logs():
    """Test that sensitive configuration is masked when logged."""
    from hipaa_compliance_summarizer.config import mask_sensitive_config
    
    config_with_secrets = {
        "epic_api_key": "sk-very-secret-key-12345",  # pragma: allowlist secret
        "database_url": "postgresql://user:password123@localhost/db",  # pragma: allowlist secret
        "encryption_key": "base64-encoded-encryption-key",  # pragma: allowlist secret
        "non_secret": "this is not secret"  # pragma: allowlist secret
    }
    
    masked = mask_sensitive_config(config_with_secrets)
    
    # Secrets should be masked
    assert masked["epic_api_key"] == "sk-ver***"  # pragma: allowlist secret
    assert "password123" not in masked["database_url"]
    assert masked["encryption_key"] == "base64***"  # pragma: allowlist secret
    # Non-secrets should remain unchanged
    assert masked["non_secret"] == "this is not secret"  # pragma: allowlist secret


def test_config_integration_with_hipaa_processor():
    """Test that HIPAAProcessor can use environment-based configuration."""
    from hipaa_compliance_summarizer.processor import HIPAAProcessor
    
    test_env = {
        "EPIC_API_KEY": "test_integration_key",  # pragma: allowlist secret
        "COMPLIANCE_LEVEL": "strict"
    }
    
    with patch.dict(os.environ, test_env):
        processor = HIPAAProcessor()
        
        # Processor should be able to access environment configuration
        # This tests the integration without requiring actual implementation yet
        assert processor.compliance_level is not None


def test_environment_config_overrides_file_config():
    """Test that environment variables override file-based configuration."""
    from hipaa_compliance_summarizer.config import load_config
    
    # Test that environment variables take precedence
    file_config = "compliance:\n  level: standard\napi_key: file_key"
    env_override = {
        "HIPAA_CONFIG_YAML": file_config,
        "EPIC_API_KEY": "env_override_key"  # pragma: allowlist secret
    }
    
    with patch.dict(os.environ, env_override):
        config = load_config()
        
        # File config should be loaded
        assert config.get("compliance", {}).get("level") == "standard"
        
        # But environment should override sensitive values
        from hipaa_compliance_summarizer.config import get_secret_config
        secrets = get_secret_config()
        assert secrets.get("epic_api_key") == "env_override_key"  # pragma: allowlist secret