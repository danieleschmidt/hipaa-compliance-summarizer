from hipaa_compliance_summarizer.phi import PHIRedactor


def test_redact_file(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("SSN: 123-45-6789")
    redactor = PHIRedactor()
    res = redactor.redact_file(str(f))
    assert "[REDACTED]" in res.text
    assert res.entities[0].type == "ssn"
