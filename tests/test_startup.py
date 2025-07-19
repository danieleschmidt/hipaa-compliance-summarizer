import os
import pytest
from unittest.mock import patch
import logging


def test_validate_environment_development():
    """Test environment validation for development mode."""
    from hipaa_compliance_summarizer.startup import validate_environment
    
    # Test development environment (less strict)
    result = validate_environment(require_production_secrets=False)
    
    # Should pass even without secrets
    assert result["valid"] is True
    assert "config" in result
    assert isinstance(result["errors"], list)
    assert isinstance(result["warnings"], list)


def test_validate_environment_production():
    """Test environment validation for production mode."""
    from hipaa_compliance_summarizer.startup import validate_environment
    
    # Test production environment (strict)
    with patch.dict(os.environ, {}, clear=True):
        result = validate_environment(require_production_secrets=True)
        
        # Should fail without required secrets
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("encryption_key" in error.lower() for error in result["errors"])


def test_validate_environment_production_with_secrets():
    """Test environment validation for production with proper secrets."""
    from hipaa_compliance_summarizer.startup import validate_environment
    
    production_env = {
        "ENCRYPTION_KEY": "test_encryption_key_123",
        "DATABASE_URL": "postgresql://user:pass@localhost/db",  # pragma: allowlist secret
        "EPIC_API_KEY": "test_epic_key"  # pragma: allowlist secret
    }
    
    with patch.dict(os.environ, production_env):
        result = validate_environment(require_production_secrets=True)
        
        # Should pass with proper configuration
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["config"]["encryption_key"] == "test_encryption_key_123"


def test_check_required_environment_for_production_success():
    """Test production environment check with valid configuration."""
    from hipaa_compliance_summarizer.startup import check_required_environment_for_production
    
    production_env = {
        "ENCRYPTION_KEY": "prod_encryption_key",
        "DATABASE_URL": "postgresql://user:pass@prod-db/hipaa"  # pragma: allowlist secret
    }
    
    with patch.dict(os.environ, production_env):
        config = check_required_environment_for_production()
        assert config["encryption_key"] == "prod_encryption_key"


def test_check_required_environment_for_production_failure():
    """Test production environment check with invalid configuration."""
    from hipaa_compliance_summarizer.startup import check_required_environment_for_production
    
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            check_required_environment_for_production()
        
        assert "Production environment validation failed" in str(exc_info.value)


def test_setup_logging_with_config():
    """Test logging setup with environment configuration."""
    from hipaa_compliance_summarizer.startup import setup_logging_with_config
    
    test_env = {
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "%(levelname)s: %(message)s"
    }
    
    with patch.dict(os.environ, test_env):
        # Reset logging to allow basicConfig to work
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.NOTSET)
        
        setup_logging_with_config()
        
        # Verify logging level was set (check root logger since basicConfig affects it)
        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() <= logging.DEBUG


def test_get_environment_info():
    """Test getting environment information."""
    from hipaa_compliance_summarizer.startup import get_environment_info
    
    test_env = {
        "ENVIRONMENT": "test",
        "APP_VERSION": "1.0.0",
        "DEBUG": "true"
    }
    
    with patch.dict(os.environ, test_env):
        info = get_environment_info()
        
        assert info["environment"] == "test"
        assert info["app_version"] == "1.0.0"
        assert info["debug_mode"] is True


def test_environment_info_defaults():
    """Test environment information with default values."""
    from hipaa_compliance_summarizer.startup import get_environment_info
    
    with patch.dict(os.environ, {}, clear=True):
        info = get_environment_info()
        
        assert info["environment"] == "development"
        assert info["app_version"] == "0.0.1" 
        assert info["debug_mode"] is False