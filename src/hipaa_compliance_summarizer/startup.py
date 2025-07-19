"""Startup utilities for configuration validation and environment setup."""

import logging
import os
from typing import Dict, Any

from .config import get_secret_config, validate_secret_config, mask_sensitive_config

logger = logging.getLogger(__name__)


def validate_environment(require_production_secrets: bool = False) -> Dict[str, Any]:
    """Validate environment configuration at startup.
    
    Args:
        require_production_secrets: If True, enforce production-level validation
        
    Returns:
        Dictionary containing validation results and loaded configuration
        
    Raises:
        RuntimeError: If critical configuration errors are found
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": {}
    }
    
    try:
        # Load secret configuration from environment
        secret_config = get_secret_config()
        result["config"] = secret_config
        
        # Validate configuration
        errors = validate_secret_config(secret_config, required_for_production=require_production_secrets)
        result["errors"] = errors
        
        # Log configuration status (masked for security)
        masked_config = mask_sensitive_config(secret_config)
        logger.info("Loaded configuration: %s", masked_config)
        
        # Check for common issues
        if require_production_secrets:
            if not secret_config.get("encryption_key"):
                result["errors"].append("ENCRYPTION_KEY environment variable is required for production")
            
            if not secret_config.get("database_url"):
                result["warnings"].append("DATABASE_URL not configured - using in-memory storage")
        
        # Set validation status based on actual errors, not warnings
        critical_errors = [error for error in result["errors"] if "Required secret" in error]
        if critical_errors:
            result["valid"] = False
            logger.error("Configuration validation failed: %s", critical_errors)
        else:
            result["valid"] = True
            if result["errors"]:  # These are warnings, not critical errors
                result["warnings"].extend(result["errors"])
                result["errors"] = critical_errors
            
            if result["warnings"]:
                logger.warning("Configuration warnings: %s", result["warnings"])
            else:
                logger.info("Configuration validation passed")
            
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Configuration loading failed: {str(e)}")
        logger.exception("Failed to load configuration")
    
    return result


def check_required_environment_for_production():
    """Check that required environment variables are set for production deployment.
    
    Raises:
        RuntimeError: If required production environment variables are missing
    """
    validation = validate_environment(require_production_secrets=True)
    
    if not validation["valid"]:
        error_msg = "Production environment validation failed:\n" + "\n".join(validation["errors"])
        raise RuntimeError(error_msg)
    
    return validation["config"]


def setup_logging_with_config():
    """Setup logging configuration based on environment variables."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format
    )
    
    # Suppress noisy loggers in production
    if os.environ.get("ENVIRONMENT") == "production":
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    logger.info("Logging configured with level: %s", log_level)


def get_environment_info() -> Dict[str, str]:
    """Get information about the current environment.
    
    Returns:
        Dictionary with environment information for debugging
    """
    return {
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
        "app_version": os.environ.get("APP_VERSION", "0.0.1"),
        "deployment_region": os.environ.get("DEPLOYMENT_REGION", "local"),
        "debug_mode": os.environ.get("DEBUG", "false").lower() == "true"
    }