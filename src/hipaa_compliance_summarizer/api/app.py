"""FastAPI application factory for HIPAA compliance API."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from ..database import get_db_connection, initialize_database
from ..monitoring import setup_metrics_middleware
from .middleware import (
    audit_middleware,
    auth_middleware,
    rate_limit_middleware,
    security_headers_middleware,
)
from .routes import api_blueprint

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HIPAA Compliance API...")

    # Initialize database
    try:
        db = initialize_database()
        if db.test_connection():
            logger.info("Database connection established")
        else:
            logger.error("Database connection failed")
            raise RuntimeError("Failed to connect to database")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    # Initialize monitoring
    try:
        setup_metrics_middleware(app)
        logger.info("Monitoring initialized")
    except Exception as e:
        logger.warning(f"Monitoring setup failed: {e}")

    logger.info("HIPAA Compliance API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down HIPAA Compliance API...")
    logger.info("HIPAA Compliance API shutdown complete")


def create_app(config: dict = None) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured FastAPI application
    """
    # Application configuration
    app_config = config or {}
    debug_mode = app_config.get("debug", os.getenv("DEBUG", "false").lower() == "true")

    # Create FastAPI app
    app = FastAPI(
        title="HIPAA Compliance Summarizer API",
        description="Healthcare-focused API for automated PHI detection, redaction, and compliance reporting",
        version="1.2.0",
        debug=debug_mode,
        lifespan=lifespan,
        docs_url="/docs" if debug_mode else None,  # Disable docs in production
        redoc_url="/redoc" if debug_mode else None,
        openapi_url="/openapi.json" if debug_mode else None,
    )

    # Security middleware
    setup_security_middleware(app, app_config)

    # CORS middleware
    setup_cors_middleware(app, app_config)

    # Custom middleware
    setup_custom_middleware(app)

    # Include API routes
    app.include_router(api_blueprint, prefix="/api/v1")

    # Health check endpoint
    @app.get("/health", include_in_schema=False)
    async def health_check():
        """Health check endpoint."""
        try:
            # Test database connection
            db = get_db_connection()
            db_healthy = db.test_connection()

            return {
                "status": "healthy" if db_healthy else "unhealthy",
                "database": "connected" if db_healthy else "disconnected",
                "version": "1.2.0"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "version": "1.2.0"
                }
            )

    # Metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        # In production, this would return Prometheus-formatted metrics
        return {"metrics": "placeholder"}

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="HIPAA Compliance Summarizer API",
            version="1.2.0",
            description="""
# HIPAA Compliance Summarizer API

Healthcare-focused API for automated PHI detection, redaction, and compliance reporting.

## Features

- **PHI Detection**: Automated identification of 18 HIPAA-defined PHI categories
- **Document Processing**: Support for clinical notes, lab reports, insurance forms
- **Compliance Reporting**: Audit-ready compliance reports and metrics
- **Security**: End-to-end encryption, audit logging, access controls
- **Batch Processing**: High-throughput processing for enterprise volumes

## Authentication

All endpoints require valid API authentication. Include your API key in the `Authorization` header:

```
Authorization: Bearer your-api-key-here
```

## Rate Limiting

API requests are rate-limited to protect system resources:
- 100 requests per minute per API key
- 1000 requests per hour per API key
- Burst allowance of 20 requests

## Compliance

This API is designed for HIPAA compliance:
- All PHI is encrypted at rest and in transit
- Comprehensive audit logging for all operations
- 7-year data retention for compliance requirements
- SOC 2 Type II certified infrastructure

## Support

For technical support or compliance questions:
- Email: compliance@hipaa-summarizer.com
- Documentation: https://docs.hipaa-summarizer.com
            """,
            routes=app.routes,
        )

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "API key authentication"
            }
        }

        # Apply security to all endpoints
        for path in openapi_schema["paths"]:
            for method in openapi_schema["paths"][path]:
                if method != "options":
                    openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


def setup_security_middleware(app: FastAPI, config: dict):
    """Setup security middleware."""
    # Trusted hosts
    allowed_hosts = config.get("allowed_hosts", ["localhost", "127.0.0.1"])
    if os.getenv("ENVIRONMENT") == "production":
        # In production, be more restrictive
        allowed_hosts = config.get("production_hosts", ["api.hipaa-summarizer.com"])

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )

    # Security headers
    app.middleware("http")(security_headers_middleware)


def setup_cors_middleware(app: FastAPI, config: dict):
    """Setup CORS middleware."""
    cors_origins = config.get("cors_origins", [])

    # Default origins for development
    if not cors_origins and os.getenv("ENVIRONMENT") != "production":
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
        )


def setup_custom_middleware(app: FastAPI):
    """Setup custom middleware."""
    # Rate limiting
    app.middleware("http")(rate_limit_middleware)

    # Authentication
    app.middleware("http")(auth_middleware)

    # Audit logging
    app.middleware("http")(audit_middleware)
