"""Startup utilities for configuration validation and environment setup."""

import logging
import os
from typing import Dict, Any

from .config import get_secret_config, validate_secret_config, mask_sensitive_config
from .logging_framework import setup_structured_logging, get_logger_with_metrics, LoggingConfig

logger = get_logger_with_metrics(__name__)


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
        logger.info("Configuration loaded successfully", {
            "config": masked_config,
            "require_production_secrets": require_production_secrets
        })
        
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
            logger.error("Configuration validation failed", {
                "critical_errors": critical_errors,
                "error_count": len(critical_errors)
            })
        else:
            result["valid"] = True
            if result["errors"]:  # These are warnings, not critical errors
                result["warnings"].extend(result["errors"])
                result["errors"] = critical_errors
            
            if result["warnings"]:
                logger.warning("Configuration warnings detected", {
                    "warnings": result["warnings"],
                    "warning_count": len(result["warnings"])
                })
            else:
                logger.info("Configuration validation passed successfully")
            
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Configuration loading failed: {str(e)}")
        logger.error("Failed to load configuration", {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=e)
    
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
    """Setup structured logging configuration based on environment variables."""
    config = LoggingConfig.from_environment()
    metrics_collector = setup_structured_logging(config)
    
    # Suppress noisy loggers in production
    if os.environ.get("ENVIRONMENT") == "production":
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    logger.info("Structured logging configured successfully", {
        "log_level": config.level,
        "output_format": config.output_format,
        "metrics_enabled": config.enable_metrics,
        "environment": os.environ.get("ENVIRONMENT", "development")
    })
    
    return metrics_collector


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