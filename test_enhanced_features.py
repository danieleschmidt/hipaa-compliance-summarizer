#!/usr/bin/env python3
"""Test enhanced security, performance, and ML features."""

import time
from src.hipaa_compliance_summarizer.security_enhanced import get_security_manager
from src.hipaa_compliance_summarizer.performance_enhanced import get_performance_optimizer, performance_monitor
from src.hipaa_compliance_summarizer.ml_integration_lite import get_ml_processor

def test_security_features():
    """Test security manager functionality."""
    security_manager = get_security_manager()
    
    # Test security event logging
    event_id = security_manager.log_security_event(
        "test_event", "LOW", details={"test": True}
    )
    assert event_id is not None
    print(f"✅ Security event logged: {event_id}")
    
    # Test access attempt tracking
    result = security_manager.track_access_attempt("192.168.1.1", success=True)
    assert result is True
    print("✅ Access attempt tracking works")
    
    # Test session token generation
    token = security_manager.generate_session_token("test_user")
    assert security_manager.validate_session_token(token) is True
    print("✅ Session token generation and validation works")
    
    # Test security dashboard
    dashboard = security_manager.get_security_dashboard()
    assert "timestamp" in dashboard
    print("✅ Security dashboard generation works")

def test_performance_features():
    """Test performance optimizer functionality."""
    perf_optimizer = get_performance_optimizer()
    
    # Test performance monitoring decorator
    @performance_monitor("test_operation")
    def test_operation():
        time.sleep(0.1)
        return "completed"
    
    result = test_operation()
    assert result == "completed"
    print("✅ Performance monitoring decorator works")
    
    # Test cache functionality
    cache = perf_optimizer.cache
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    assert value == "test_value"
    print("✅ Adaptive cache works")
    
    # Test performance report
    report = perf_optimizer.get_performance_report(hours=1)
    assert "total_operations" in report
    print("✅ Performance report generation works")

def test_ml_features():
    """Test ML processor functionality."""
    ml_processor = get_ml_processor()
    
    # Test PHI prediction
    test_text = "Patient John Smith, DOB: 01/01/1980, SSN: 123-45-6789"
    prediction = ml_processor.predict_phi_entities(test_text)
    assert prediction.confidence > 0
    print(f"✅ PHI prediction works (confidence: {prediction.confidence})")
    
    # Test compliance scoring
    compliance_prediction = ml_processor.calculate_compliance_score(test_text, 3)
    assert isinstance(compliance_prediction.prediction, float)
    print(f"✅ Compliance scoring works (score: {compliance_prediction.prediction})")
    
    # Test document similarity
    documents = ["Patient has diabetes", "Patient diagnosed with diabetes mellitus"]
    similarity = ml_processor.analyze_document_similarity(documents)
    assert similarity.confidence > 0
    print("✅ Document similarity analysis works")

def run_all_tests():
    """Run all enhanced feature tests."""
    print("Testing Enhanced Security Features...")
    test_security_features()
    print()
    
    print("Testing Enhanced Performance Features...")
    test_performance_features()
    print()
    
    print("Testing Enhanced ML Features...")
    test_ml_features()
    print()
    
    print("🎉 All enhanced features working correctly!")

if __name__ == "__main__":
    run_all_tests()