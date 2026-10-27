from scripts.check_setup import check_config


def test_placeholder_key_rejected():
    assert check_config({"OPENAI_API_KEY": "请填写你的API密钥"})


def test_missing_local_model_rejected(tmp_path):
    assert check_config({
        "OPENAI_API_KEY": "test-key",
        "EMBEDDING_PROVIDER": "local",
        "LOCAL_EMBEDDING_MODEL": str(tmp_path / "missing"),
    })


def test_model_identifier_allowed():
    assert not check_config({
        "OPENAI_API_KEY": "test-key",
        "EMBEDDING_PROVIDER": "local",
        "LOCAL_EMBEDDING_MODEL": "BAAI/bge-small-zh-v1.5",
    })
