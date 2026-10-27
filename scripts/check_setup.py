"""启动前检查：不加载模型、不调用远程服务，也不输出密钥。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


def check_config(values: dict) -> list[str]:
    errors = []
    key = str(values.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "sk-placeholder" or "填写" in key or "粘贴" in key:
        errors.append("请在 .env 中填写自己的 OPENAI_API_KEY。")
    if values.get("EMBEDDING_PROVIDER") == "local":
        model = str(values.get("LOCAL_EMBEDDING_MODEL") or "BAAI/bge-small-zh-v1.5")
        if (Path(model).is_absolute() or model.startswith(".")) and not Path(model).exists():
            errors.append("LOCAL_EMBEDDING_MODEL 指定的本地模型目录不存在，请修改路径。")
    return errors


def main() -> int:
    errors = []
    if not (ROOT / ".env").exists():
        errors.append("尚未配置：请复制 .env.example 为 .env，填写密钥后重试。")
    else:
        errors.extend(check_config(dict(dotenv_values(ROOT / ".env"))))
    for name in ["fastapi", "uvicorn", "chromadb", "langchain_openai"]:
        if importlib.util.find_spec(name) is None:
            errors.append(f"缺少依赖 {name}，请先运行 setup.ps1。")
    if errors:
        for error in errors:
            print(f"[需要处理] {error}")
        return 1
    print("配置检查通过。接下来加载向量模型；首次下载可能较慢，请等待启动完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
