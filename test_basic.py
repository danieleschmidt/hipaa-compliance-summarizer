#!/usr/bin/env python3
"""Basic functionality test for HIPAA Compliance Summarizer"""

from hipaa_compliance_summarizer import HIPAAProcessor

def test_basic_functionality():
    """Test basic processing functionality."""
    processor = HIPAAProcessor(compliance_level="standard")
    
    # Test with demo document
    result = processor.process_document("demo_document.txt")
    
    print(f"Summary: {result.summary}")
    print(f"Compliance Score: {result.compliance_score}")
    print(f"PHI Detected: {result.phi_detected_count}")
    print(f"Processing successful: {result.compliance_score > 0}")

if __name__ == "__main__":
    test_basic_functionality()