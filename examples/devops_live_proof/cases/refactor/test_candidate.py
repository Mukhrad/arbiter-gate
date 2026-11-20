from candidate import normalize_key


def test_mechanical_normalization():
    assert normalize_key("  Gate-Config ") == "gate_config"
    assert normalize_key("MODEL") == "model"
