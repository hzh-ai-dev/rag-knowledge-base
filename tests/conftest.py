"""pytest 全局配置。

注意：环境变量必须在导入 ``app.*`` 之前设置好，
因为 ``app.config.settings`` 在模块导入时就会实例化（这是刻意的单例设计）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# 测试用的临时数据目录，避免污染开发环境的 data/
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="rag-test-"))

os.environ["DATA_DIR"] = str(_TMP_ROOT / "data")
os.environ["CHROMA_DIR"] = str(_TMP_ROOT / "chroma")
os.environ["UPLOAD_DIR"] = str(_TMP_ROOT / "uploads")
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["ENABLE_HYBRID"] = "true"
os.environ["ENABLE_RERANK"] = "false"

import pytest  # noqa: E402


@pytest.fixture()
def tmp_chroma(tmp_path) -> Path:
    """每个用例独立的向量库目录。"""
    directory = tmp_path / "chroma"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
