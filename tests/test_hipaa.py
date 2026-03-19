"""
tests/test_hipaa.py
===================
Unit tests for the HIPAA Compliance Pipeline.
One test per PHI category, plus integration tests.
"""

import json
import pytest
from hipaa_compliance import (
    PHIDetector,
    PHIRedactor,
    ComplianceAuditor,
    ComplianceSummarizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def detector():
    return PHIDetector()


@pytest.fixture
def redactor():
    return PHIRedactor()


@pytest.fixture
def auditor():
    return ComplianceAuditor()


@pytest.fixture
def summarizer():
    return ComplianceSummarizer()


# ── PHI Category Tests ────────────────────────────────────────────────────────

# 1. NAME
def test_detect_name(detector):
    # Use a text where the name isn't preceded by a title word
    text = "Admitted: John Smith to ward 3."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "NAME" in types
    values = [f["value"] for f in findings if f["type"] == "NAME"]
    assert any("John Smith" in v for v in values)


# 2. GEOGRAPHIC — street address
def test_detect_geographic_street(detector):
    text = "Lives at 123 Main St, Springfield, IL 62701."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "GEOGRAPHIC" in types


# 2b. GEOGRAPHIC — zip code
def test_detect_geographic_zip(detector):
    text = "ZIP code 90210 is in California."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "GEOGRAPHIC" in types


# 3. DATE
def test_detect_date_numeric(detector):
    text = "Admitted on 03/15/2024."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "DATE" in types


def test_detect_date_written(detector):
    text = "Born on February 14, 1990."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "DATE" in types


# 4. PHONE
def test_detect_phone(detector):
    text = "Call (555) 867-5309 for appointments."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "PHONE" in types


# 5. FAX
def test_detect_fax(detector):
    text = "Fax: (555) 867-5310."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "FAX" in types


# 6. EMAIL
def test_detect_email(detector):
    text = "Send records to jdoe@hospital.org please."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "EMAIL" in types


# 7. SSN
def test_detect_ssn(detector):
    text = "SSN: 123-45-6789."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "SSN" in types


# 8. MRN
def test_detect_mrn(detector):
    text = "MRN: 987654 — see chart."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "MRN" in types


def test_detect_mrn_long_form(detector):
    text = "Medical Record Number: 111222."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "MRN" in types


# 9. HEALTH_PLAN_BENEFICIARY
def test_detect_health_plan_beneficiary(detector):
    text = "Beneficiary ID: BEN987654."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "HEALTH_PLAN_BENEFICIARY" in types


# 10. ACCOUNT_NUMBER
def test_detect_account_number(detector):
    text = "Account Number: ACC-987654."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "ACCOUNT_NUMBER" in types


# 11. CERTIFICATE_LICENSE
def test_detect_certificate_license(detector):
    text = "License: MD123456."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "CERTIFICATE_LICENSE" in types


# 12. VEHICLE_IDENTIFIER — VIN
def test_detect_vehicle_vin(detector):
    # 17-char VIN (no I, O, Q)
    text = "VIN: 1HGBH41JXMN109186."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "VEHICLE_IDENTIFIER" in types


# 12b. VEHICLE_IDENTIFIER — plate keyword
def test_detect_vehicle_plate(detector):
    text = "Plate: ABC1234."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "VEHICLE_IDENTIFIER" in types


# 13. DEVICE_IDENTIFIER
def test_detect_device_identifier(detector):
    text = "Device S/N: DEV12345."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "DEVICE_IDENTIFIER" in types


# 14. URL
def test_detect_url(detector):
    text = "Visit https://hospital.org/records for details."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "URL" in types


# 15. IP_ADDRESS
def test_detect_ip_address(detector):
    text = "Accessed from IP 192.168.1.100."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "IP_ADDRESS" in types


# 16. BIOMETRIC
def test_detect_biometric(detector):
    text = "Fingerprint ID: FP-7823."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "BIOMETRIC" in types


# 17. PHOTO_REFERENCE
def test_detect_photo_reference(detector):
    text = "Photo: patient_headshot.jpg was uploaded."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "PHOTO_REFERENCE" in types


# 18. AGE_OVER_89
def test_detect_age_over_89(detector):
    text = "The 95 year old patient was examined."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "AGE_OVER_89" in types


def test_detect_age_over_89_high(detector):
    text = "Patient is 102 years old."
    findings = detector.detect(text)
    types = [f["type"] for f in findings]
    assert "AGE_OVER_89" in types


def test_no_age_under_90(detector):
    """Ages 89 and below should NOT be flagged."""
    text = "The 45 year old patient presented."
    findings = detector.detect(text)
    age_types = [f for f in findings if f["type"] == "AGE_OVER_89"]
    assert len(age_types) == 0


# ── Redactor Tests ─────────────────────────────────────────────────────────────

def test_redactor_replaces_phi(detector, redactor):
    text = "Patient John Smith, SSN 123-45-6789."
    findings = detector.detect(text)
    redacted = redactor.redact(text, findings)
    assert "John Smith" not in redacted
    assert "123-45-6789" not in redacted
    assert "[NAME]" in redacted or "[SSN]" in redacted


def test_redactor_empty_findings(redactor):
    text = "No PHI here."
    assert redactor.redact(text, []) == text


def test_redactor_overlap_handling(redactor):
    """Overlapping spans should not cause errors."""
    text = "Test 12345 overlap."
    findings = [
        {"type": "A", "value": "12345", "start": 5, "end": 10},
        {"type": "B", "value": "1234", "start": 5, "end": 9},   # overlaps A
    ]
    result = redactor.redact(text, findings)
    assert "12345" not in result or "[A]" in result or "[B]" in result


# ── Auditor Tests ──────────────────────────────────────────────────────────────

def test_auditor_records_entries(detector, auditor):
    text = "Patient John Smith, SSN 123-45-6789."
    findings = detector.detect(text)
    auditor.record("doc1", findings, redacted=True)
    assert len(auditor.log) == len(findings)
    for entry in auditor.log:
        assert entry["document_id"] == "doc1"
        assert entry["redacted"] is True
        assert "timestamp" in entry
        assert "phi_type" in entry
        assert "phi_value" in entry


def test_auditor_summary_by_type(detector, auditor):
    text = "John Smith called (555) 867-5309."
    findings = detector.detect(text)
    auditor.record("doc2", findings)
    summary = auditor.summary_by_type()
    assert isinstance(summary, dict)
    assert all(isinstance(v, int) for v in summary.values())


def test_auditor_to_json(detector, auditor):
    text = "SSN 123-45-6789."
    findings = detector.detect(text)
    auditor.record("doc3", findings)
    payload = json.loads(auditor.to_json())
    assert isinstance(payload, list)
    assert len(payload) > 0


def test_auditor_clear(detector, auditor):
    text = "SSN 123-45-6789."
    findings = detector.detect(text)
    auditor.record("doc4", findings)
    auditor.clear()
    assert auditor.log == []


# ── Summarizer Tests ───────────────────────────────────────────────────────────

def test_summarizer_high_risk(detector, auditor, summarizer):
    text = "SSN 123-45-6789."
    findings = detector.detect(text)
    auditor.record("doc_high", findings, redacted=True)
    report = summarizer.summarize(auditor.log, text, text)
    assert report["risk_level"] == "HIGH"


def test_summarizer_medium_risk(detector, auditor, summarizer):
    text = "Patient John Smith called (555) 867-5309."
    findings = detector.detect(text)
    auditor.record("doc_med", findings, redacted=True)
    report = summarizer.summarize(auditor.log, text, text)
    assert report["risk_level"] in ("HIGH", "MEDIUM")


def test_summarizer_redaction_score(detector, auditor, summarizer, redactor):
    text = "Patient John Smith, SSN 123-45-6789."
    findings = detector.detect(text)
    redacted = redactor.redact(text, findings)
    auditor.record("doc_score", findings, redacted=True)
    report = summarizer.summarize(auditor.log, text, redacted)
    assert report["redaction_score"] == 100.0


def test_summarizer_empty_log(summarizer):
    report = summarizer.summarize([], "clean text", "clean text")
    assert report["risk_level"] == "NONE"
    assert report["phi_count"] == 0
    assert report["redaction_score"] == 100.0


def test_summarizer_phi_categories_list(detector, auditor, summarizer):
    text = "Email jdoe@hospital.org, phone (555) 234-5678."
    findings = detector.detect(text)
    auditor.record("doc_cat", findings, redacted=True)
    report = summarizer.summarize(auditor.log, text, text)
    assert "EMAIL" in report["phi_categories"]
    assert "PHONE" in report["phi_categories"]


# ── Integration Test ───────────────────────────────────────────────────────────

def test_full_pipeline():
    """End-to-end: detect → redact → audit → summarize."""
    text = (
        "Patient John Smith, DOB 03/15/1965, SSN 123-45-6789, "
        "MRN: 987654, admitted 01/10/2024. Call (555) 234-5678."
    )
    det = PHIDetector()
    red = PHIRedactor()
    aud = ComplianceAuditor()
    summ = ComplianceSummarizer()

    findings = det.detect(text)
    assert len(findings) > 0

    redacted = red.redact(text, findings)
    assert "John Smith" not in redacted
    assert "123-45-6789" not in redacted

    aud.record("integration_test", findings, redacted=True)
    report = summ.summarize(aud.log, text, redacted)

    assert report["risk_level"] == "HIGH"
    assert report["redaction_score"] == 100.0
    assert "SSN" in report["phi_categories"]
    assert "MRN" in report["phi_categories"]
