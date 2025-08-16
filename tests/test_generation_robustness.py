"""Comprehensive tests for Generation 2 robustness features."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from hipaa_compliance_summarizer.validation import (
    InputValidator, ValidationLevel, ValidationResult, validate_input_with_context
)
from hipaa_compliance_summarizer.audit_logger import (
    AuditLogger, AuditEventType, AuditLevel, get_audit_logger
)
from hipaa_compliance_summarizer.error_handling import (
    ErrorHandler, HIPAAError, ValidationError, ErrorCategory, ErrorSeverity
)


class TestInputValidation:
    """Test comprehensive input validation system."""

    def test_text_validation_basic(self):
        """Test basic text input validation."""
        validator = InputValidator(ValidationLevel.STANDARD)
        
        # Valid text
        result = validator.validate_text_input("Valid medical text")
        assert result.is_valid
        assert result.sanitized_input == "Valid medical text"
        assert len(result.errors) == 0
        
        # None input
        result = validator.validate_text_input(None)
        assert not result.is_valid
        assert "cannot be None" in result.errors[0]

    def test_text_validation_malicious_patterns(self):
        """Test detection of malicious patterns in text."""
        validator = InputValidator(ValidationLevel.STRICT)
        
        # Script injection
        result = validator.validate_text_input("<script>alert('xss')</script>Patient data")
        assert not result.is_valid
        assert any("Malicious patterns" in error for error in result.errors)
        
        # JavaScript protocol
        result = validator.validate_text_input("Click javascript:alert('xss') for info")
        assert not result.is_valid

    def test_text_validation_length_limits(self):
        """Test text length validation."""
        validator = InputValidator(ValidationLevel.STANDARD)
        
        # Text too long
        long_text = "x" * 10000000  # 10MB
        result = validator.validate_text_input(long_text, max_length=1000)
        assert not result.is_valid
        assert "too long" in result.errors[0]

    def test_file_path_validation(self):
        """Test file path validation."""
        validator = InputValidator(ValidationLevel.STANDARD)
        
        # Valid path
        result = validator.validate_file_path("/tmp/valid_file.txt")
        assert result.is_valid
        
        # Path traversal attack
        result = validator.validate_file_path("../../../etc/passwd")
        assert result.risk_score > 0.5
        
        # Windows path traversal
        result = validator.validate_file_path("..\\..\\windows\\system32")
        assert result.risk_score > 0.5

    def test_compliance_level_validation(self):
        """Test compliance level validation."""
        validator = InputValidator(ValidationLevel.STANDARD)
        
        # Valid string inputs
        for level in ["strict", "standard", "minimal"]:
            result = validator.validate_compliance_level(level)
            assert result.is_valid
            assert result.sanitized_input == level
        
        # Invalid input
        result = validator.validate_compliance_level("invalid")
        assert not result.is_valid
        
        # None input
        result = validator.validate_compliance_level(None)
        assert not result.is_valid

    def test_numeric_validation(self):
        """Test numeric input validation."""
        validator = InputValidator(ValidationLevel.STANDARD)
        
        # Valid numeric inputs
        result = validator.validate_numeric_input(42.5)
        assert result.is_valid
        assert result.sanitized_input == 42.5
        
        # String numeric
        result = validator.validate_numeric_input("123.45")
        assert result.is_valid
        assert result.sanitized_input == 123.45
        
        # Range validation
        result = validator.validate_numeric_input(150, min_val=0, max_val=100)
        assert not result.is_valid or result.sanitized_input == 100
        
        # Invalid numeric string
        result = validator.validate_numeric_input("123abc")
        assert not result.is_valid

    def test_configuration_validation(self):
        """Test configuration dictionary validation."""
        validator = InputValidator(ValidationLevel.STANDARD)
        
        # Valid configuration
        config = {
            "compliance_level": "strict",
            "max_workers": 4,
            "timeout": 30.0,
            "patterns": ["pattern1", "pattern2"]
        }
        result = validator.validate_configuration_dict(config)
        assert result.is_valid
        
        # Configuration with malicious content
        malicious_config = {
            "script": "<script>alert('xss')</script>",
            "command": "rm -rf /",
            "normal_key": "normal_value"
        }
        result = validator.validate_configuration_dict(malicious_config)
        assert len(result.warnings) > 0

    def test_validation_with_context(self):
        """Test validation with context wrapper."""
        # Valid input
        result = validate_input_with_context("Valid text", "text", "test_context")
        assert result == "Valid text"
        
        # Invalid input should raise ValidationError
        with pytest.raises(ValidationError):
            validate_input_with_context(None, "text", "test_context")


class TestAuditLogging:
    """Test comprehensive audit logging system."""

    def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            assert logger.log_file == log_file
            assert logger.session_id is not None
            assert logger.event_count >= 1  # Session start event

    def test_basic_event_logging(self):
        """Test basic event logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            # Log a test event
            event_id = logger.log_event(
                event_type=AuditEventType.DOCUMENT_PROCESSING,
                level=AuditLevel.INFO,
                operation="test_operation",
                user_id="test_user",
                document_id="doc_123"
            )
            
            assert event_id is not None
            assert len(event_id) > 0
            assert logger.event_count >= 2  # Session start + test event

    def test_document_processing_logging(self):
        """Test document processing specific logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            event_id = logger.log_document_processing(
                document_id="doc_123",
                operation="process_medical_record",
                duration_ms=1500.0,
                phi_detected=5,
                compliance_score=0.95,
                status="success",
                user_id="user_456"
            )
            
            assert event_id is not None
            assert AuditEventType.DOCUMENT_PROCESSING in logger.events_by_type

    def test_security_event_logging(self):
        """Test security event logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            event_id = logger.log_security_event(
                event_description="Suspicious file upload attempt",
                severity=AuditLevel.WARNING,
                ip_address="192.168.1.100",
                user_id="user_123",
                details={"file_type": "executable", "blocked": True}
            )
            
            assert event_id is not None
            assert AuditEventType.SECURITY_EVENT in logger.events_by_type

    def test_phi_detection_logging(self):
        """Test PHI detection logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            event_id = logger.log_phi_detection(
                document_id="doc_789",
                phi_type="SSN",
                confidence=0.98,
                redaction_method="replacement",
                user_id="user_789"
            )
            
            assert event_id is not None
            assert AuditEventType.PHI_DETECTION in logger.events_by_type

    def test_compliance_check_logging(self):
        """Test compliance check logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            violations = ["Missing patient consent", "Inadequate encryption"]
            event_id = logger.log_compliance_check(
                document_id="doc_456",
                compliance_level="strict",
                score=0.75,
                violations=violations,
                user_id="admin_user"
            )
            
            assert event_id is not None
            assert AuditEventType.COMPLIANCE_CHECK in logger.events_by_type

    def test_session_summary(self):
        """Test session summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            logger = AuditLogger(log_file=log_file)
            
            # Log some events
            logger.log_event(AuditEventType.USER_ACTION, operation="login")
            logger.log_event(AuditEventType.DOCUMENT_PROCESSING, operation="process")
            
            summary = logger.get_session_summary()
            
            assert "session_id" in summary
            assert "total_events" in summary
            assert summary["total_events"] >= 3
            assert "events_by_type" in summary

    def test_audit_trail_export(self):
        """Test audit trail export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_audit.log"
            export_file = Path(temp_dir) / "exported_audit.json"
            logger = AuditLogger(log_file=log_file)
            
            # Log some events
            logger.log_event(AuditEventType.DOCUMENT_PROCESSING, operation="process1")
            logger.log_event(AuditEventType.SECURITY_EVENT, operation="security_check")
            
            # Export audit trail
            events = logger.export_audit_trail(
                event_types=[AuditEventType.DOCUMENT_PROCESSING],
                output_file=export_file
            )
            
            assert len(events) >= 1
            assert export_file.exists()
            
            # Verify exported content
            with open(export_file, 'r') as f:
                exported_data = json.load(f)
            assert len(exported_data) >= 1

    def test_global_audit_logger(self):
        """Test global audit logger singleton."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        
        assert logger1 is logger2  # Should be the same instance


class TestErrorHandling:
    """Test advanced error handling system."""

    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler()
        
        assert handler.error_history == []
        assert isinstance(handler.retry_strategies, dict)
        assert isinstance(handler.error_callbacks, dict)

    def test_basic_error_handling(self):
        """Test basic error handling functionality."""
        handler = ErrorHandler()
        
        # Create a test error context
        from hipaa_compliance_summarizer.error_handling import ErrorContext
        import datetime
        
        context = ErrorContext(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.datetime.utcnow(),
            source="test",
            operation="test_operation"
        )
        
        # Handle a standard exception
        test_exception = ValueError("Test error")
        result = handler.handle_error(test_exception, context, auto_retry=False)
        
        assert len(handler.error_history) == 1
        assert isinstance(handler.error_history[0], HIPAAError)

    def test_retry_strategy_registration(self):
        """Test retry strategy registration and execution."""
        handler = ErrorHandler()
        
        # Register a test retry strategy
        def test_retry_strategy(error):
            return "retry_successful"
        
        handler.register_retry_strategy("TEST_ERROR", test_retry_strategy)
        assert "TEST_ERROR" in handler.retry_strategies

    def test_error_callback_registration(self):
        """Test error callback registration and execution."""
        handler = ErrorHandler()
        callback_executed = False
        
        def test_callback(error):
            nonlocal callback_executed
            callback_executed = True
        
        handler.register_error_callback(ErrorCategory.SECURITY, test_callback)
        
        # Create and handle a security error
        from hipaa_compliance_summarizer.error_handling import ErrorContext
        import datetime
        
        context = ErrorContext(
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            timestamp=datetime.datetime.utcnow(),
            source="test",
            operation="security_test"
        )
        
        test_exception = RuntimeError("Security test error")
        handler.handle_error(test_exception, context, auto_retry=False)
        
        assert callback_executed

    def test_error_statistics(self):
        """Test error statistics generation."""
        handler = ErrorHandler()
        
        # Create multiple errors
        from hipaa_compliance_summarizer.error_handling import ErrorContext
        import datetime
        
        for i in range(3):
            context = ErrorContext(
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.MEDIUM,
                timestamp=datetime.datetime.utcnow(),
                source="test",
                operation=f"test_operation_{i}"
            )
            handler.handle_error(ValueError(f"Test error {i}"), context, auto_retry=False)
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert "by_category" in stats
        assert "by_severity" in stats
        assert "recent_errors" in stats

    def test_hipaa_error_creation(self):
        """Test HIPAA error creation and serialization."""
        from hipaa_compliance_summarizer.error_handling import ErrorContext
        import datetime
        
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            timestamp=datetime.datetime.utcnow(),
            source="test",
            operation="validation_test",
            user_id="test_user",
            document_id="test_doc"
        )
        
        error = HIPAAError(
            message="Test HIPAA error",
            error_code="TEST_ERROR_001",
            context=context
        )
        
        # Test serialization
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "TEST_ERROR_001"
        assert error_dict["message"] == "Test HIPAA error"
        assert error_dict["category"] == "validation"
        assert error_dict["severity"] == "high"
        assert error_dict["user_id"] == "test_user"

    def test_validation_error_handling(self):
        """Test validation-specific error handling."""
        # Test ValidationError creation
        from hipaa_compliance_summarizer.error_handling import ErrorContext
        import datetime
        
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.datetime.utcnow(),
            source="input_validator",
            operation="validate_text"
        )
        
        validation_error = ValidationError(
            message="Input validation failed",
            error_code="VALIDATION_TEXT_FAILED",
            context=context
        )
        
        assert isinstance(validation_error, HIPAAError)
        assert validation_error.context.category == ErrorCategory.VALIDATION


class TestIntegrationRobustness:
    """Test integration between robustness components."""

    def test_validation_with_audit_logging(self):
        """Test integration between validation and audit logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "integration_audit.log"
            audit_logger = AuditLogger(log_file=log_file)
            
            # Perform validation that should trigger audit logging
            validator = InputValidator(ValidationLevel.STRICT)
            
            # This should generate warnings that could be audited
            result = validator.validate_text_input("<script>malicious</script>Safe content")
            
            # Manually log validation results
            if not result.is_valid:
                audit_logger.log_security_event(
                    event_description="Malicious input detected during validation",
                    severity=AuditLevel.WARNING,
                    details={
                        "risk_score": result.risk_score,
                        "errors": result.errors,
                        "warnings": result.warnings
                    }
                )
            
            assert AuditEventType.SECURITY_EVENT in audit_logger.events_by_type

    def test_error_handling_with_audit_logging(self):
        """Test integration between error handling and audit logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "error_audit.log"
            audit_logger = AuditLogger(log_file=log_file)
            error_handler = ErrorHandler()
            
            # Register callback to log errors to audit system
            def audit_error_callback(error):
                audit_logger.log_error(
                    error_type=type(error).__name__,
                    error_message=error.message,
                    operation=error.context.operation,
                    document_id=error.context.document_id,
                    details={"error_code": error.error_code}
                )
            
            error_handler.register_error_callback(ErrorCategory.PROCESSING, audit_error_callback)
            
            # Create and handle an error
            from hipaa_compliance_summarizer.error_handling import ErrorContext
            import datetime
            
            context = ErrorContext(
                category=ErrorCategory.PROCESSING,
                severity=ErrorSeverity.HIGH,
                timestamp=datetime.datetime.utcnow(),
                source="test",
                operation="integration_test",
                document_id="test_doc_123"
            )
            
            test_exception = RuntimeError("Integration test error")
            error_handler.handle_error(test_exception, context, auto_retry=False)
            
            # Verify audit logging occurred
            assert AuditEventType.ERROR_EVENT in audit_logger.events_by_type

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_validation_under_resource_pressure(self, mock_cpu, mock_memory):
        """Test validation behavior under resource pressure."""
        # Simulate high memory usage
        mock_memory.return_value.percent = 90.0
        mock_cpu.return_value = 85.0
        
        validator = InputValidator(ValidationLevel.STANDARD)
        
        # Validation should still work but might be more conservative
        large_text = "x" * 1000000  # 1MB text
        result = validator.validate_text_input(large_text)
        
        # Should handle large input gracefully
        assert result is not None

    def test_comprehensive_error_recovery(self):
        """Test comprehensive error recovery scenarios."""
        handler = ErrorHandler()
        recovery_count = 0
        
        def recovery_strategy(error):
            nonlocal recovery_count
            recovery_count += 1
            return f"recovered_{recovery_count}"
        
        handler.register_retry_strategy("PROCESSING_RUNTIMEERROR", recovery_strategy)
        
        # Create error context
        from hipaa_compliance_summarizer.error_handling import ErrorContext
        import datetime
        
        context = ErrorContext(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.datetime.utcnow(),
            source="test",
            operation="recovery_test"
        )
        
        # Handle error with recovery
        test_exception = RuntimeError("Recoverable error")
        result = handler.handle_error(test_exception, context, auto_retry=True)
        
        assert result == "recovered_1"
        assert recovery_count == 1