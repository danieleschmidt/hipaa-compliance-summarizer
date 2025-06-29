from hipaa_compliance_summarizer.phi import PHIRedactor


def test_pattern_cache():
    r1 = PHIRedactor().patterns["ssn"]
    r2 = PHIRedactor().patterns["ssn"]
    assert r1 is r2
