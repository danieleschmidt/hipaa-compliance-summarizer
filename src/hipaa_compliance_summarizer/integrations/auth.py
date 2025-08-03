"""Authentication and authorization integrations."""

import os
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib
import secrets
from urllib.parse import urlencode
import base64

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT token payload structure."""
    
    user_id: str
    email: str
    roles: List[str]
    permissions: List[str]
    session_id: str
    issued_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT encoding."""
        return {
            "sub": self.user_id,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "session_id": self.session_id,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()),
            "iss": "hipaa-compliance-summarizer",
            "aud": "hipaa-api"
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPayload":
        """Create from JWT decoded payload."""
        return cls(
            user_id=data["sub"],
            email=data["email"],
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            session_id=data["session_id"],
            issued_at=datetime.fromtimestamp(data["iat"]),
            expires_at=datetime.fromtimestamp(data["exp"])
        )


class JWTManager:
    """JWT token management for API authentication."""
    
    def __init__(self, secret_key: str = None, algorithm: str = "HS256"):
        """Initialize JWT manager.
        
        Args:
            secret_key: JWT signing secret key
            algorithm: JWT signing algorithm
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET") or self._generate_secret()
        self.algorithm = algorithm
        self.token_expiry_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        
        if not secret_key and not os.getenv("JWT_SECRET"):
            logger.warning("No JWT secret provided - using generated secret (not suitable for production)")
    
    def _generate_secret(self) -> str:
        """Generate a secure random secret key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(64)).decode()
    
    def create_token(self, user_id: str, email: str, roles: List[str] = None,
                    permissions: List[str] = None, session_id: str = None) -> str:
        """Create a JWT token for a user.
        
        Args:
            user_id: Unique user identifier
            email: User email address
            roles: User roles
            permissions: User permissions
            session_id: Optional session ID
            
        Returns:
            Encoded JWT token
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.token_expiry_hours)
        
        payload = TokenPayload(
            user_id=user_id,
            email=email,
            roles=roles or [],
            permissions=permissions or [],
            session_id=session_id or secrets.token_urlsafe(32),
            issued_at=now,
            expires_at=expires_at
        )
        
        try:
            token = jwt.encode(payload.to_dict(), self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created JWT token for user {user_id}")
            return token
        except Exception as e:
            logger.error(f"Failed to create JWT token: {e}")
            raise
    
    def decode_token(self, token: str) -> TokenPayload:
        """Decode and validate a JWT token.
        
        Args:
            token: JWT token to decode
            
        Returns:
            Decoded token payload
            
        Raises:
            jwt.InvalidTokenError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload.from_dict(payload)
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise
    
    def refresh_token(self, token: str) -> str:
        """Refresh an existing JWT token.
        
        Args:
            token: Current JWT token
            
        Returns:
            New JWT token with extended expiry
        """
        try:
            payload = self.decode_token(token)
            
            # Create new token with same user info but new expiry
            return self.create_token(
                user_id=payload.user_id,
                email=payload.email,
                roles=payload.roles,
                permissions=payload.permissions,
                session_id=payload.session_id
            )
        except Exception as e:
            logger.error(f"Failed to refresh JWT token: {e}")
            raise
    
    def validate_token(self, token: str) -> bool:
        """Validate if a JWT token is valid and not expired.
        
        Args:
            token: JWT token to validate
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            self.decode_token(token)
            return True
        except jwt.InvalidTokenError:
            return False
    
    def revoke_session(self, session_id: str):
        """Revoke all tokens for a specific session.
        
        Note: In production, maintain a blacklist of revoked sessions
        """
        # This would typically involve adding the session to a Redis blacklist
        logger.info(f"Session {session_id} revoked")
    
    def get_user_permissions(self, token: str) -> List[str]:
        """Extract user permissions from token.
        
        Args:
            token: JWT token
            
        Returns:
            List of user permissions
        """
        try:
            payload = self.decode_token(token)
            return payload.permissions
        except jwt.InvalidTokenError:
            return []


class OAuthHandler:
    """OAuth 2.0 authentication handler for external integrations."""
    
    def __init__(self):
        """Initialize OAuth handler."""
        self.client_configs = {
            "google": {
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
                "scope": "openid email profile",
                "auth_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo"
            },
            "azure": {
                "client_id": os.getenv("AZURE_CLIENT_ID"),
                "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
                "redirect_uri": os.getenv("AZURE_REDIRECT_URI"),
                "scope": "openid email profile",
                "tenant_id": os.getenv("AZURE_TENANT_ID"),
                "auth_url": f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}/oauth2/v2.0/authorize",
                "token_url": f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}/oauth2/v2.0/token",
                "userinfo_url": "https://graph.microsoft.com/v1.0/me"
            }
        }
        
        self.jwt_manager = JWTManager()
    
    def get_authorization_url(self, provider: str, state: str = None) -> str:
        """Generate OAuth authorization URL.
        
        Args:
            provider: OAuth provider (google, azure)
            state: Optional state parameter for CSRF protection
            
        Returns:
            Authorization URL
        """
        if provider not in self.client_configs:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.client_configs[provider]
        
        if not all([config["client_id"], config["redirect_uri"]]):
            raise ValueError(f"OAuth configuration incomplete for {provider}")
        
        params = {
            "client_id": config["client_id"],
            "redirect_uri": config["redirect_uri"],
            "scope": config["scope"],
            "response_type": "code",
            "access_type": "offline"
        }
        
        if state:
            params["state"] = state
        
        auth_url = f"{config['auth_url']}?{urlencode(params)}"
        logger.info(f"Generated OAuth URL for {provider}")
        
        return auth_url
    
    def exchange_code_for_token(self, provider: str, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token.
        
        Args:
            provider: OAuth provider
            code: Authorization code from OAuth callback
            
        Returns:
            Token response with access_token and user info
        """
        if provider not in self.client_configs:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.client_configs[provider]
        
        # In production, make actual HTTP request to token endpoint
        # For now, return mock response
        logger.info(f"Exchanging OAuth code for {provider}")
        
        # Mock user info (replace with actual OAuth implementation)
        user_info = {
            "id": f"{provider}_user_123",
            "email": "user@example.com",
            "name": "Test User",
            "provider": provider
        }
        
        # Create internal JWT token
        jwt_token = self.jwt_manager.create_token(
            user_id=user_info["id"],
            email=user_info["email"],
            roles=["user"],
            permissions=["read", "write"]
        )
        
        return {
            "access_token": jwt_token,
            "token_type": "Bearer",
            "expires_in": self.jwt_manager.token_expiry_hours * 3600,
            "user_info": user_info
        }
    
    def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Get user information from OAuth provider.
        
        Args:
            provider: OAuth provider
            access_token: Access token
            
        Returns:
            User information
        """
        if provider not in self.client_configs:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        # In production, make HTTP request to userinfo endpoint
        # For now, extract from our JWT token
        try:
            payload = self.jwt_manager.decode_token(access_token)
            return {
                "id": payload.user_id,
                "email": payload.email,
                "name": payload.email.split("@")[0],  # Mock name
                "provider": provider
            }
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            raise
    
    def revoke_token(self, provider: str, token: str):
        """Revoke an OAuth token.
        
        Args:
            provider: OAuth provider
            token: Token to revoke
        """
        try:
            payload = self.jwt_manager.decode_token(token)
            self.jwt_manager.revoke_session(payload.session_id)
            logger.info(f"Revoked {provider} token for session {payload.session_id}")
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")


class APIKeyManager:
    """API key management for service-to-service authentication."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.api_keys = {}  # In production, use database or Redis
    
    def generate_api_key(self, service_name: str, permissions: List[str] = None) -> str:
        """Generate a new API key for a service.
        
        Args:
            service_name: Name of the service
            permissions: List of permissions for the API key
            
        Returns:
            Generated API key
        """
        api_key = f"hipaa_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "service_name": service_name,
            "permissions": permissions or [],
            "created_at": datetime.utcnow(),
            "last_used": None,
            "active": True
        }
        
        logger.info(f"Generated API key for service: {service_name}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate an API key and return associated metadata.
        
        Args:
            api_key: API key to validate
            
        Returns:
            API key metadata if valid
            
        Raises:
            ValueError: If API key is invalid or inactive
        """
        if api_key not in self.api_keys:
            raise ValueError("Invalid API key")
        
        key_info = self.api_keys[api_key]
        
        if not key_info["active"]:
            raise ValueError("API key is inactive")
        
        # Update last used timestamp
        key_info["last_used"] = datetime.utcnow()
        
        return key_info
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key.
        
        Args:
            api_key: API key to revoke
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info("API key revoked")
        else:
            logger.warning("Attempted to revoke non-existent API key")
    
    def list_api_keys(self, service_name: str = None) -> List[Dict[str, Any]]:
        """List API keys, optionally filtered by service.
        
        Args:
            service_name: Optional service name filter
            
        Returns:
            List of API key metadata (excluding the actual keys)
        """
        keys = []
        for key, info in self.api_keys.items():
            if service_name is None or info["service_name"] == service_name:
                key_info = info.copy()
                key_info["key_hash"] = hashlib.sha256(key.encode()).hexdigest()[:16]
                keys.append(key_info)
        
        return keys