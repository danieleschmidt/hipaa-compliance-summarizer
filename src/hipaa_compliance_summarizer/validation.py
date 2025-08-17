"""Comprehensive input validation and sanitization for HIPAA compliance system."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

from .constants import PROCESSING_CONSTANTS, SECURITY_LIMITS
from .error_handling import ErrorCategory, ErrorContext, ErrorSeverity, ValidationError

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Input validation strictness levels."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Any
    warnings: List[str]
    errors: List[str]
    risk_score: float


class InputValidator:
    """Comprehensive input validation and sanitization system."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript protocol
            r'data:.*base64',             # Data URLs
            r'eval\s*\(',                 # JavaScript eval
            r'exec\s*\(',                 # Python exec
            r'__import__\s*\(',           # Python imports
            r'subprocess\.',              # Subprocess calls
            r'os\.',                      # OS module calls
            r'\.\./',                     # Path traversal
            r'\x00',                      # Null bytes
            r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
        ]
        self.suspicious_keywords = [
            'password', 'secret', 'token', 'key', 'admin', 'root',
            'drop table', 'delete from', 'insert into', 'update set',
            'union select', 'or 1=1', 'and 1=1', '--', ';--',
        ]

    def validate_text_input(self, text: Any, max_length: Optional[int] = None) -> ValidationResult:
        """Validate and sanitize text input."""
        warnings = []
        errors = []
        risk_score = 0.0

        # Type validation
        if text is None:
            errors.append("Input cannot be None")
            return ValidationResult(False, "", warnings, errors, 1.0)

        if not isinstance(text, str):
            try:
                text = str(text)
                warnings.append(f"Input converted from {type(text).__name__} to string")
            except Exception as e:
                errors.append(f"Cannot convert input to string: {e}")
                return ValidationResult(False, "", warnings, errors, 1.0)

        # Length validation
        max_len = max_length or SECURITY_LIMITS.MAX_DOCUMENT_SIZE
        if len(text) > max_len:
            errors.append(f"Input too long: {len(text)} characters (max {max_len})")
            return ValidationResult(False, text[:max_len], warnings, errors, 0.8)

        # Unicode validation and normalization
        try:
            text = unicodedata.normalize('NFKC', text)
        except UnicodeError as e:
            errors.append(f"Invalid Unicode content: {e}")
            return ValidationResult(False, "", warnings, errors, 1.0)

        # Check for malicious patterns
        malicious_found = []
        for pattern in self.malicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                malicious_found.extend(matches)
                risk_score += 0.2

        if malicious_found:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Malicious patterns detected: {malicious_found}")
                return ValidationResult(False, "", warnings, errors, min(risk_score, 1.0))
            else:
                warnings.append(f"Potentially malicious patterns detected and removed: {len(malicious_found)}")
                for pattern in self.malicious_patterns:
                    text = re.sub(pattern, '[SANITIZED]', text, flags=re.IGNORECASE | re.DOTALL)

        # Check for suspicious keywords
        suspicious_found = []
        for keyword in self.suspicious_keywords:
            if keyword.lower() in text.lower():
                suspicious_found.append(keyword)
                risk_score += 0.1

        if suspicious_found:
            warnings.append(f"Suspicious keywords detected: {suspicious_found}")

        # Control character filtering
        control_chars = [char for char in text
                        if ord(char) < PROCESSING_CONSTANTS.CONTROL_CHAR_THRESHOLD
                        and char not in '\t\n\r']

        if control_chars:
            warnings.append(f"Removed {len(control_chars)} control characters")
            text = ''.join(char for char in text
                          if ord(char) >= PROCESSING_CONSTANTS.CONTROL_CHAR_THRESHOLD
                          or char in '\t\n\r')

        # Null byte check
        if '\x00' in text:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append("Null bytes detected in input")
                return ValidationResult(False, "", warnings, errors, 1.0)
            else:
                warnings.append("Null bytes removed from input")
                text = text.replace('\x00', '')

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=text,
            warnings=warnings,
            errors=errors,
            risk_score=min(risk_score, 1.0)
        )

    def validate_file_path(self, path: Union[str, Path]) -> ValidationResult:
        """Validate file path for security."""
        warnings = []
        errors = []
        risk_score = 0.0

        if path is None:
            errors.append("File path cannot be None")
            return ValidationResult(False, "", warnings, errors, 1.0)

        path_str = str(path)

        # Length check
        if len(path_str) > SECURITY_LIMITS.MAX_PATH_LENGTH:
            errors.append(f"Path too long: {len(path_str)} characters")
            return ValidationResult(False, "", warnings, errors, 0.8)

        # Dangerous pattern checks
        dangerous_patterns = [
            r'\.\./', r'\.\.\\',  # Path traversal
            r'~/', r'/etc/', r'/proc/', r'/sys/',  # System paths
            r'\\\\', r'///',  # Network paths
            r'[<>:"|?*]',  # Invalid Windows characters
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, path_str, re.IGNORECASE):
                risk_score += 0.3
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"Dangerous path pattern detected: {pattern}")

        # Check for executable extensions
        executable_exts = {'.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.app'}
        path_obj = Path(path_str)
        if path_obj.suffix.lower() in executable_exts:
            risk_score += 0.5
            warnings.append(f"Executable file extension detected: {path_obj.suffix}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=Path(path_str),
            warnings=warnings,
            errors=errors,
            risk_score=min(risk_score, 1.0)
        )

    def validate_compliance_level(self, level: Any) -> ValidationResult:
        """Validate compliance level input."""
        warnings = []
        errors = []

        if level is None:
            errors.append("Compliance level cannot be None")
            return ValidationResult(False, "standard", warnings, errors, 0.5)

        # Handle string inputs
        if isinstance(level, str):
            level_map = {
                'strict': 'strict',
                'standard': 'standard',
                'minimal': 'minimal'
            }
            normalized = level.lower().strip()
            if normalized in level_map:
                return ValidationResult(True, level_map[normalized], warnings, errors, 0.0)
            else:
                errors.append(f"Invalid compliance level: '{level}'. Must be strict/standard/minimal")
                return ValidationResult(False, "standard", warnings, errors, 0.3)

        # Handle other types
        try:
            level_str = str(level).lower().strip()
            if level_str in ['strict', 'standard', 'minimal']:
                return ValidationResult(True, level_str, warnings, errors, 0.0)
            else:
                errors.append(f"Invalid compliance level: {level}")
                return ValidationResult(False, "standard", warnings, errors, 0.3)
        except Exception as e:
            errors.append(f"Cannot process compliance level: {e}")
            return ValidationResult(False, "standard", warnings, errors, 0.5)

    def validate_numeric_input(self, value: Any, min_val: float = None, max_val: float = None) -> ValidationResult:
        """Validate numeric input with range checking."""
        warnings = []
        errors = []
        risk_score = 0.0

        if value is None:
            errors.append("Numeric value cannot be None")
            return ValidationResult(False, 0, warnings, errors, 0.5)

        # Type conversion
        try:
            if isinstance(value, (int, float)):
                numeric_value = float(value)
            elif isinstance(value, str):
                # Check for injection attempts in numeric strings
                if re.search(r'[^\d\.\-\+e]', value):
                    errors.append("Invalid characters in numeric string")
                    return ValidationResult(False, 0, warnings, errors, 0.8)
                numeric_value = float(value)
            else:
                errors.append(f"Cannot convert {type(value).__name__} to numeric value")
                return ValidationResult(False, 0, warnings, errors, 0.5)
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid numeric value: {e}")
            return ValidationResult(False, 0, warnings, errors, 0.5)

        # Range validation
        if min_val is not None and numeric_value < min_val:
            errors.append(f"Value {numeric_value} below minimum {min_val}")
            return ValidationResult(False, min_val, warnings, errors, 0.3)

        if max_val is not None and numeric_value > max_val:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Value {numeric_value} above maximum {max_val}")
                return ValidationResult(False, max_val, warnings, errors, 0.3)
            else:
                warnings.append(f"Value {numeric_value} clamped to maximum {max_val}")
                numeric_value = max_val

        # Check for suspicious values
        if abs(numeric_value) > 1e10:
            warnings.append("Extremely large numeric value detected")
            risk_score += 0.1

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=numeric_value,
            warnings=warnings,
            errors=errors,
            risk_score=risk_score
        )

    def validate_configuration_dict(self, config: Any) -> ValidationResult:
        """Validate configuration dictionary."""
        warnings = []
        errors = []
        risk_score = 0.0

        if config is None:
            return ValidationResult(True, {}, warnings, errors, 0.0)

        if not isinstance(config, dict):
            try:
                config = dict(config)
                warnings.append("Configuration converted to dictionary")
            except Exception as e:
                errors.append(f"Cannot convert configuration to dictionary: {e}")
                return ValidationResult(False, {}, warnings, errors, 0.5)

        # Validate each key-value pair
        sanitized_config = {}
        for key, value in config.items():
            # Validate key
            key_result = self.validate_text_input(str(key), max_length=100)
            if not key_result.is_valid:
                warnings.append(f"Invalid configuration key '{key}': {key_result.errors}")
                continue

            # Validate value based on type
            if isinstance(value, str):
                value_result = self.validate_text_input(value, max_length=1000)
                if value_result.is_valid:
                    sanitized_config[key_result.sanitized_input] = value_result.sanitized_input
                else:
                    warnings.append(f"Invalid value for key '{key}': {value_result.errors}")
                    risk_score += 0.1
            elif isinstance(value, (int, float)):
                value_result = self.validate_numeric_input(value)
                if value_result.is_valid:
                    sanitized_config[key_result.sanitized_input] = value_result.sanitized_input
                else:
                    warnings.append(f"Invalid numeric value for key '{key}': {value_result.errors}")
            elif isinstance(value, (list, tuple)):
                # Validate list/tuple elements
                sanitized_list = []
                for item in value:
                    if isinstance(item, str):
                        item_result = self.validate_text_input(item, max_length=500)
                        if item_result.is_valid:
                            sanitized_list.append(item_result.sanitized_input)
                    else:
                        sanitized_list.append(item)
                sanitized_config[key_result.sanitized_input] = sanitized_list
            else:
                # For other types, store as-is but log warning
                warnings.append(f"Unvalidated value type for key '{key}': {type(value).__name__}")
                sanitized_config[key_result.sanitized_input] = value

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized_config,
            warnings=warnings,
            errors=errors,
            risk_score=min(risk_score, 1.0)
        )


def validate_input_with_context(
    input_value: Any,
    input_type: str,
    context: str = "unknown",
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> Any:
    """Validate input with comprehensive context and error handling."""
    validator = InputValidator(validation_level)

    try:
        if input_type == "text":
            result = validator.validate_text_input(input_value)
        elif input_type == "file_path":
            result = validator.validate_file_path(input_value)
        elif input_type == "compliance_level":
            result = validator.validate_compliance_level(input_value)
        elif input_type == "numeric":
            result = validator.validate_numeric_input(input_value)
        elif input_type == "config":
            result = validator.validate_configuration_dict(input_value)
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        # Log warnings
        for warning in result.warnings:
            logger.warning(f"Input validation warning in {context}: {warning}")

        # Handle errors
        if not result.is_valid:
            error_context = ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.HIGH if result.risk_score > 0.7 else ErrorSeverity.MEDIUM,
                timestamp=logger.getEffectiveLevel(),  # Using logger level as timestamp substitute
                source="input_validator",
                operation=f"validate_{input_type}",
                additional_context={
                    "input_type": input_type,
                    "context": context,
                    "risk_score": result.risk_score,
                    "errors": result.errors
                }
            )

            raise ValidationError(
                message=f"Input validation failed in {context}: {'; '.join(result.errors)}",
                error_code=f"VALIDATION_{input_type.upper()}_FAILED",
                context=error_context
            )

        # Log high-risk inputs
        if result.risk_score > 0.5:
            logger.warning(
                f"High-risk input detected in {context}: risk_score={result.risk_score}, "
                f"warnings={len(result.warnings)}"
            )

        return result.sanitized_input

    except ValidationError:
        raise
    except Exception as e:
        error_context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            timestamp=logger.getEffectiveLevel(),
            source="input_validator",
            operation=f"validate_{input_type}",
            additional_context={
                "input_type": input_type,
                "context": context,
                "original_error": str(e)
            }
        )

        raise ValidationError(
            message=f"Validation system error in {context}: {e}",
            error_code="VALIDATION_SYSTEM_ERROR",
            context=error_context,
            original_exception=e
        )


__all__ = [
    "ValidationLevel",
    "ValidationResult",
    "InputValidator",
    "validate_input_with_context"
]
