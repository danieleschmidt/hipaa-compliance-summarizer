"""Security utilities for input validation and sanitization."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, List
import logging

from .constants import SECURITY_LIMITS

logger = logging.getLogger(__name__)

# Security constants (imported from centralized config)
MAX_FILE_SIZE = SECURITY_LIMITS.MAX_FILE_SIZE
MAX_PATH_LENGTH = SECURITY_LIMITS.MAX_PATH_LENGTH
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.html', '.xml', '.json', '.csv'}
BLOCKED_PATTERNS = [
    r'\.\./',  # Path traversal
    r'\.\.\\',  # Windows path traversal
    r'~/',     # Home directory access
    r'/etc/',  # System files
    r'/proc/', # Process files
    r'/sys/',  # System files
]


class SecurityError(Exception):
    """Raised when a security validation fails.
    
    Attributes:
        file_path: Optional file path that caused the security error
        violation_type: Type of security violation detected
        original_error: Original exception that triggered this security error
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 violation_type: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.file_path = file_path
        self.violation_type = violation_type
        self.original_error = original_error
    
    def get_context(self) -> dict:
        """Get contextual information about the security error."""
        context = {}
        if self.file_path:
            context["file_path"] = self.file_path
        if self.violation_type:
            context["violation_type"] = self.violation_type
        if self.original_error:
            context["original_error"] = str(self.original_error)
        return context


def validate_file_path(file_path: str) -> Path:
    """Validate and sanitize a file path for security.
    
    Args:
        file_path: The file path to validate
        
    Returns:
        Resolved Path object if valid
        
    Raises:
        SecurityError: If the path is invalid or potentially dangerous
    """
    if not file_path or not isinstance(file_path, str):
        raise SecurityError("File path must be a non-empty string")
    
    # Check path length
    if len(file_path) > MAX_PATH_LENGTH:
        raise SecurityError(f"File path too long (max {MAX_PATH_LENGTH} characters)")
    
    # Check for blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, file_path, re.IGNORECASE):
            raise SecurityError(f"Potentially dangerous path pattern detected: {pattern}")
    
    # Convert to Path and resolve
    try:
        path = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise SecurityError(f"Invalid file path: {e}")
    
    # Ensure the path doesn't escape expected boundaries
    if '..' in str(path):
        raise SecurityError("Path traversal detected in resolved path")
    
    return path


def validate_file_size(file_path: Path) -> None:
    """Validate that a file size is within acceptable limits.
    
    Args:
        file_path: Path to the file to check
        
    Raises:
        SecurityError: If the file is too large or inaccessible
    """
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise SecurityError(
                f"File too large: {file_size} bytes (max {MAX_FILE_SIZE} bytes)"
            )
        
        if file_size == 0:
            logger.warning("Empty file detected: %s", file_path)
            
    except OSError as e:
        raise SecurityError(f"Cannot access file: {e}")


def validate_file_extension(file_path: Path) -> None:
    """Validate that a file has an allowed extension.
    
    Args:
        file_path: Path to the file to check
        
    Raises:
        SecurityError: If the file extension is not allowed
    """
    extension = file_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise SecurityError(
            f"File extension '{extension}' not allowed. "
            f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def validate_directory_path(dir_path: str) -> Path:
    """Validate and sanitize a directory path for security.
    
    Args:
        dir_path: The directory path to validate
        
    Returns:
        Resolved Path object if valid
        
    Raises:
        SecurityError: If the path is invalid or potentially dangerous
    """
    validated_path = validate_file_path(dir_path)
    
    if not validated_path.exists():
        raise SecurityError(f"Directory does not exist: {validated_path}")
    
    if not validated_path.is_dir():
        raise SecurityError(f"Path is not a directory: {validated_path}")
    
    # Check directory permissions
    if not os.access(validated_path, os.R_OK):
        raise SecurityError(f"Directory is not readable: {validated_path}")
    
    return validated_path


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing potentially dangerous characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "unknown_file"
    
    # Remove path separators and other dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "unknown_file"
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        max_name_len = 255 - len(ext)
        sanitized = name[:max_name_len] + ext
    
    return sanitized


def validate_content_type(file_path: Path) -> bool:
    """Basic content type validation by examining file signature.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the content appears safe, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(512)  # Read first 512 bytes
        
        # Check for executable file signatures
        executable_signatures = [
            b'MZ',      # PE executable
            b'\x7fELF', # ELF executable
            b'\xca\xfe\xba\xbe',  # Mach-O universal binary
            b'#!/bin/', # Shell script
            b'#!/usr/bin/',  # Shell script
        ]
        
        for signature in executable_signatures:
            if header.startswith(signature):
                logger.warning("Potentially executable file detected: %s", file_path)
                return False
        
        # Check for script content in text files
        if file_path.suffix.lower() in {'.txt', '.html', '.xml'}:
            try:
                text_content = header.decode('utf-8', errors='ignore').lower()
                dangerous_keywords = ['<script', 'javascript:', 'vbscript:', 'onload=']
                
                for keyword in dangerous_keywords:
                    if keyword in text_content:
                        logger.warning("Potentially dangerous script content in: %s", file_path)
                        return False
            except UnicodeDecodeError as e:
                logger.debug("Cannot decode file as UTF-8, treating as binary: %s", file_path)
                # Binary file, skip text checks - this is expected and safe
        
        return True
        
    except OSError as e:
        logger.error("Cannot read file for content validation: %s", e)
        return False


def validate_file_for_processing(file_path: str) -> Path:
    """Comprehensive file validation for HIPAA document processing.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If any validation check fails
    """
    # Step 1: Validate and sanitize path
    validated_path = validate_file_path(file_path)
    
    # Step 2: Check if file exists and is readable
    if not validated_path.exists():
        raise SecurityError(f"File does not exist: {validated_path}")
    
    if not validated_path.is_file():
        raise SecurityError(f"Path is not a regular file: {validated_path}")
    
    if not os.access(validated_path, os.R_OK):
        raise SecurityError(f"File is not readable: {validated_path}")
    
    # Step 3: Validate file size
    validate_file_size(validated_path)
    
    # Step 4: Validate file extension
    validate_file_extension(validated_path)
    
    # Step 5: Basic content type validation
    if not validate_content_type(validated_path):
        raise SecurityError(f"File content appears potentially dangerous: {validated_path}")
    
    logger.info("File validation successful: %s", validated_path)
    return validated_path


def get_security_recommendations() -> List[str]:
    """Get a list of security recommendations for deployment."""
    return [
        "Run the application with minimal required privileges",
        "Use a dedicated user account with restricted permissions",
        "Implement file system isolation (chroot/containers)",
        "Enable comprehensive audit logging",
        "Regularly update dependencies for security patches",
        "Implement rate limiting for file processing",
        "Use virus scanning on uploaded files",
        "Encrypt sensitive data at rest and in transit",
        "Implement proper session management",
        "Use HTTPS for all web communications",
    ]


__all__ = [
    "SecurityError",
    "validate_file_path",
    "validate_file_size", 
    "validate_file_extension",
    "validate_directory_path",
    "sanitize_filename",
    "validate_content_type",
    "validate_file_for_processing",
    "get_security_recommendations",
]