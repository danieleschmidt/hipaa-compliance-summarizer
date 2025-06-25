from hipaa_compliance_summarizer import (
    HIPAAProcessor,
    ProcessingResult,
    Document,
    DocumentType,
)


def test_process_clinical_note(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Patient is stable")
    doc = Document(str(f), DocumentType.CLINICAL_NOTE)
    proc = HIPAAProcessor()
    result = proc.process_document(doc)
    assert isinstance(result, ProcessingResult)


def test_unknown_type_fallback(tmp_path):
    f = tmp_path / "misc.txt"
    f.write_text("Some text content")
    doc = Document(str(f), DocumentType.UNKNOWN)
    proc = HIPAAProcessor()
    result = proc.process_document(doc)
    assert isinstance(result, ProcessingResult)
