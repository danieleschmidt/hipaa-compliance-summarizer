from hipaa_compliance_summarizer import HIPAAProcessor, Document, DocumentType
import logging


def test_processor_redacts_and_scores(tmp_path):
    """Ensure end-to-end processing yields deterministic results."""

    path = tmp_path / "phi.txt"
    path.write_text("Patient SSN 123-45-6789 is stable.")

    doc = Document(str(path), DocumentType.CLINICAL_NOTE)
    proc = HIPAAProcessor()
    result = proc.process_document(doc)

    assert result.phi_detected_count == 1
    assert "[REDACTED]" in result.redacted.text
    assert 0.0 <= result.compliance_score <= 1.0


def test_processor_logs_metrics(tmp_path, caplog):
    path = tmp_path / "phi.txt"
    path.write_text("Patient SSN 123-45-6789 is stable.")
    doc = Document(str(path), DocumentType.CLINICAL_NOTE)
    proc = HIPAAProcessor()
    with caplog.at_level(logging.INFO):
        proc.process_document(doc)
    assert any("score" in rec.message for rec in caplog.records)
