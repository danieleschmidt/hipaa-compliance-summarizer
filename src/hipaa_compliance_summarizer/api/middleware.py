"""API middleware for authentication, authorization, and auditing."""

import time
import logging
import uuid
from typing import Dict, Any
import json

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..database import get_db_connection
from ..database.repositories import AuditRepository
from ..database.models import AuditRecord

logger = logging.getLogger(__name__)

# In-memory rate limiting store (use Redis in production)
rate_limit_store: Dict[str, Dict[str, Any]] = {}


async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
    
    # HIPAA compliance headers
    response.headers["X-HIPAA-Compliant"] = "true"
    response.headers["X-PHI-Protected"] = "true"
    
    # Remove server information
    response.headers.pop("Server", None)
    
    return response


async def auth_middleware(request: Request, call_next):
    """Authentication and authorization middleware."""
    # Skip auth for health checks and public endpoints
    public_paths = ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
    if any(request.url.path.startswith(path) for path in public_paths):
        return await call_next(request)
    
    # Extract authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error_code": "MISSING_AUTHORIZATION",
                "error_message": "Authorization header is required",
                "timestamp": time.time()
            }
        )
    
    # Validate bearer token format
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error_code": "INVALID_TOKEN_FORMAT",
                "error_message": "Authorization header must be in format 'Bearer <token>'",
                "timestamp": time.time()
            }
        )
    
    # Extract token
    token = auth_header[7:]  # Remove "Bearer " prefix
    
    # Validate token (simplified - use proper JWT validation in production)
    if not _validate_api_token(token):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error_code": "INVALID_TOKEN",
                "error_message": "Invalid or expired authentication token",
                "timestamp": time.time()
            }
        )
    
    # Extract user information from token
    user_info = _extract_user_info(token)
    
    # Add user context to request
    request.state.user_id = user_info.get("user_id", "unknown")
    request.state.user_role = user_info.get("role", "user")
    request.state.session_id = user_info.get("session_id", str(uuid.uuid4()))
    
    # Check authorization for sensitive endpoints
    if not _check_authorization(request.url.path, request.method, user_info):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "success": False,
                "error_code": "INSUFFICIENT_PERMISSIONS",
                "error_message": "Insufficient permissions for this operation",
                "timestamp": time.time()
            }
        )
    
    return await call_next(request)


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Get client identifier (API key or IP address)
    client_id = _get_client_id(request)
    
    # Check rate limits
    if not _check_rate_limit(client_id, request.url.path):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "error_code": "RATE_LIMIT_EXCEEDED",
                "error_message": "Rate limit exceeded. Please try again later.",
                "timestamp": time.time(),
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    limit_info = _get_rate_limit_info(client_id)
    response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(limit_info["reset_time"])
    
    return response


async def audit_middleware(request: Request, call_next):
    """Audit logging middleware."""
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Capture request start time
    start_time = time.time()
    
    # Extract request information
    user_id = getattr(request.state, "user_id", None)
    session_id = getattr(request.state, "session_id", None)
    ip_address = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("User-Agent", "unknown")
    
    # Process request
    try:
        response = await call_next(request)
        success = 200 <= response.status_code < 400
        error_message = None
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        success = False
        error_message = str(e)
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "error_message": "Internal server error",
                "request_id": request_id,
                "timestamp": time.time()
            }
        )
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Add request ID to response
    response.headers["X-Request-ID"] = request_id
    
    # Create audit record for sensitive operations
    if _should_audit_request(request.url.path, request.method):
        try:
            audit_record = AuditRecord(
                id=str(uuid.uuid4()),
                event_type=_get_event_type(request.url.path, request.method),
                event_action=request.method,
                event_description=f"{request.method} {request.url.path}",
                resource_type=_get_resource_type(request.url.path),
                resource_id=_extract_resource_id(request.url.path),
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                error_message=error_message,
                security_level=_get_security_level(request.url.path),
                compliance_relevant=True,
            )
            
            # Add additional event data
            event_data = {
                "request_id": request_id,
                "processing_time_ms": processing_time,
                "status_code": response.status_code,
                "path": str(request.url.path),
                "method": request.method,
            }
            audit_record.set_event_data(event_data)
            
            # Save audit record
            audit_repo = AuditRepository()
            audit_repo.create(audit_record)
            
        except Exception as e:
            logger.error(f"Failed to create audit record: {e}")
    
    # Log request
    logger.info(
        f"API Request - {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"User: {user_id} - "
        f"Time: {processing_time:.2f}ms - "
        f"ID: {request_id}"
    )
    
    return response


def _validate_api_token(token: str) -> bool:
    """Validate API token (simplified implementation)."""
    # In production, implement proper JWT validation
    # For now, accept any non-empty token that looks valid
    return len(token) >= 20 and token.isalnum()


def _extract_user_info(token: str) -> Dict[str, Any]:
    """Extract user information from token."""
    # In production, decode JWT and extract claims
    # For now, return mock user info
    return {
        "user_id": f"user_{token[:8]}",
        "role": "api_user",
        "session_id": str(uuid.uuid4()),
        "permissions": ["read", "write", "process"]
    }


def _check_authorization(path: str, method: str, user_info: Dict[str, Any]) -> bool:
    """Check if user is authorized for the operation."""
    # Simplified authorization logic
    # In production, implement proper RBAC
    
    permissions = user_info.get("permissions", [])
    
    # Read operations
    if method == "GET":
        return "read" in permissions
    
    # Write operations
    if method in ["POST", "PUT", "PATCH"]:
        return "write" in permissions
    
    # Delete operations
    if method == "DELETE":
        return "delete" in permissions
    
    # Processing operations
    if "process" in path:
        return "process" in permissions
    
    return True  # Allow by default for other operations


def _get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get from API key first
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return f"token_{token[:16]}"  # Use first 16 chars of token
    
    # Fall back to IP address
    return request.client.host if request.client else "unknown"


def _check_rate_limit(client_id: str, path: str) -> bool:
    """Check if client has exceeded rate limits."""
    current_time = time.time()
    
    # Initialize client data if not exists
    if client_id not in rate_limit_store:
        rate_limit_store[client_id] = {
            "requests": [],
            "last_reset": current_time
        }
    
    client_data = rate_limit_store[client_id]
    
    # Clean old requests (older than 1 minute)
    client_data["requests"] = [
        req_time for req_time in client_data["requests"]
        if current_time - req_time < 60
    ]
    
    # Check limits
    if len(client_data["requests"]) >= 100:  # 100 requests per minute
        return False
    
    # Add current request
    client_data["requests"].append(current_time)
    
    return True


def _get_rate_limit_info(client_id: str) -> Dict[str, Any]:
    """Get rate limit information for client."""
    if client_id not in rate_limit_store:
        return {"limit": 100, "remaining": 100, "reset_time": int(time.time() + 60)}
    
    client_data = rate_limit_store[client_id]
    remaining = max(0, 100 - len(client_data["requests"]))
    reset_time = int(time.time() + 60)
    
    return {
        "limit": 100,
        "remaining": remaining,
        "reset_time": reset_time
    }


def _should_audit_request(path: str, method: str) -> bool:
    """Determine if request should be audited."""
    # Audit all non-GET requests
    if method != "GET":
        return True
    
    # Audit sensitive GET requests
    sensitive_paths = [
        "/documents/",
        "/phi-detections",
        "/compliance/",
        "/batch/",
        "/system/"
    ]
    
    return any(sensitive_path in path for sensitive_path in sensitive_paths)


def _get_event_type(path: str, method: str) -> str:
    """Get event type for audit logging."""
    if "/documents" in path:
        if method == "POST":
            return "document_upload" if "/upload" in path else "document_process"
        elif method == "GET":
            return "document_access"
        elif method == "DELETE":
            return "document_delete"
    
    if "/batch" in path:
        return "batch_process"
    
    if "/compliance" in path:
        return "compliance_report"
    
    if "/phi-detections" in path:
        return "phi_access"
    
    return f"api_{method.lower()}"


def _get_resource_type(path: str) -> str:
    """Get resource type for audit logging."""
    if "/documents" in path:
        return "document"
    elif "/phi-detections" in path:
        return "phi_entity"
    elif "/compliance" in path:
        return "compliance_report"
    elif "/batch" in path:
        return "batch_process"
    elif "/system" in path:
        return "system"
    else:
        return "api"


def _extract_resource_id(path: str) -> str:
    """Extract resource ID from path."""
    # Simple extraction for paths like /documents/{id}
    parts = path.strip("/").split("/")
    for i, part in enumerate(parts):
        if part in ["documents", "phi-detections", "compliance", "batch"]:
            if i + 1 < len(parts) and parts[i + 1] not in ["upload", "process", "report"]:
                return parts[i + 1]
    return None


def _get_security_level(path: str) -> str:
    """Get security level for audit logging."""
    # PHI-related operations are sensitive
    if any(keyword in path for keyword in ["phi", "detection", "redact"]):
        return "sensitive"
    
    # System operations are critical
    if "/system" in path:
        return "critical"
    
    # Compliance operations are sensitive
    if "/compliance" in path:
        return "sensitive"
    
    return "normal"