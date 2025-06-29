from hipaa_compliance_summarizer import BatchProcessor


def test_batch_processor(tmp_path):
    (tmp_path / "a.txt").write_text("Patient: John Doe")
    (tmp_path / "b.txt").write_text("SSN: 123-45-6789")

    proc = BatchProcessor()
    results = proc.process_directory(str(tmp_path))
    assert len(results) == 2

    dashboard = proc.generate_dashboard(results)
    assert dashboard.documents_processed == 2
    assert dashboard.total_phi_detected >= 1
    dash_str = str(dashboard)
    assert "Documents processed:" in dash_str
    dash_dict = dashboard.to_dict()
    assert dash_dict["documents_processed"] == 2
    dash_json = dashboard.to_json()
    assert "documents_processed" in dash_json


def test_save_dashboard(tmp_path):
    (tmp_path / "a.txt").write_text("A")
    proc = BatchProcessor()
    results = proc.process_directory(str(tmp_path))
    out_file = tmp_path / "dash.json"
    proc.save_dashboard(results, out_file)
    assert out_file.exists()
    content = out_file.read_text()
    assert "documents_processed" in content

def test_process_directory_progress(tmp_path, capsys):
    (tmp_path / "a.txt").write_text("A")
    proc = BatchProcessor()
    proc.process_directory(str(tmp_path), show_progress=True)
    captured = capsys.readouterr()
    assert "[1/1]" in captured.out
