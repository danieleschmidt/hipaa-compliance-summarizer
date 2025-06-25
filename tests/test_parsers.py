from hipaa_compliance_summarizer.parsers import parse_medical_record


def test_parse_medical_record():
    sample_record = "Patient: John Doe\nDiagnosis: Flu"
    result = parse_medical_record(sample_record)
    assert isinstance(result, str)
    assert "John Doe" in result


def test_empty_input():
    assert parse_medical_record("") == ""
