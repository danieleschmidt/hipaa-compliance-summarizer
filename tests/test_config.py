import pytest


def test_custom_config(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """patterns:\n  foo: foo\nscoring:\n  penalty_per_entity: 1.0\n  penalty_cap: 1.0\n  strict_multiplier: 1.0\n"""
    )
    from hipaa_compliance_summarizer import config as config_mod
    import hipaa_compliance_summarizer.phi as phi_mod
    import hipaa_compliance_summarizer.processor as proc_mod

    data = config_mod.load_config(cfg)
    config_mod.CONFIG.clear()
    config_mod.CONFIG.update(data)

    redactor = phi_mod.PHIRedactor()
    assert redactor.detect("foo")[0].value == "foo"

    processor = proc_mod.HIPAAProcessor(proc_mod.ComplianceLevel.STRICT, redactor=redactor)
    result = processor.process_document("foo")
    assert pytest.approx(result.compliance_score) == 0.0

    config_mod.CONFIG.clear()


def test_env_config(monkeypatch):
    yaml_str = "patterns:\n  bar: bar\n"
    monkeypatch.setenv("HIPAA_CONFIG_YAML", yaml_str)
    from hipaa_compliance_summarizer import config as config_mod

    data = config_mod.load_config()
    assert data["patterns"]["bar"] == "bar"

    monkeypatch.delenv("HIPAA_CONFIG_YAML")
