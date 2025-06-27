import sys
from hipaa_compliance_summarizer.cli import batch_process


def test_cli_batch_process(tmp_path, capsys, monkeypatch):
    (tmp_path / "a.txt").write_text("Name: Jane Doe")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "batch_process",
        "--input-dir", str(tmp_path),
        "--output-dir", str(out_dir),
        "--show-dashboard",
        "--dashboard-json",
        str(tmp_path / "dash.json"),
    ])
    batch_process.main()
    captured = capsys.readouterr()
    assert out_dir.exists()
    assert "Documents processed:" in captured.out
    json_path = tmp_path / "dash.json"
    assert json_path.exists()


def test_cli_dashboard_json_only(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("Name: Joe")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_process",
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--dashboard-json",
            str(tmp_path / "dash.json"),
        ],
    )
    batch_process.main()
    assert out_dir.exists()
    assert (tmp_path / "dash.json").exists()
