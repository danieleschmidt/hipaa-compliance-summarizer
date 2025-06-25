from hipaa_compliance_summarizer.documents import DocumentType, detect_document_type


def test_detect_clinical_note():
    assert detect_document_type("clinical_note.txt") == DocumentType.CLINICAL_NOTE


def test_unknown_type():
    assert detect_document_type("misc.pdf") == DocumentType.UNKNOWN
