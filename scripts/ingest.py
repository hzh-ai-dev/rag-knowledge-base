"""命令行入库工具。

用法：
    python -m scripts.ingest data/samples              # 入库整个目录
    python -m scripts.ingest data/samples/产品手册.md   # 入库单个文件
    python -m scripts.ingest data/samples --reset      # 先清空再入库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许以 `python -m scripts.ingest` 或直接运行本文件两种方式执行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.container import get_ingestion_pipeline, get_vector_store  # noqa: E402
from app.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("scripts.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="将文档写入 RAG 知识库")
    parser.add_argument("source", nargs="?", default=str(settings.upload_dir),
                        help="文件或目录路径，默认 data/uploads")
    parser.add_argument("--reset", action="store_true", help="入库前先清空向量库")
    parser.add_argument("--debug", action="store_true", help="输出调试日志")
    args = parser.parse_args()

    setup_logging(args.debug)
    store = get_vector_store()

    if args.reset:
        logger.warning("清空现有向量库…")
        store.reset()

    path = Path(args.source)
    if not path.exists():
        logger.error("路径不存在: %s", path)
        return 1

    pipeline = get_ingestion_pipeline()
    if path.is_dir():
        result = pipeline.ingest_directory(path)
    else:
        result = pipeline.ingest_files([path])

    print("\n" + "=" * 52)
    print(f"  入库完成：{result.files} 个文件 / {result.chunks} 个文本块")
    print(f"  当前知识库总量：{store.count} 个文本块")
    if result.skipped:
        print(f"  跳过 {len(result.skipped)} 个：")
        for item in result.skipped:
            print(f"    - {item}")
    if result.errors:
        print(f"  失败 {len(result.errors)} 个：")
        for item in result.errors:
            print(f"    - {item}")
    print("=" * 52 + "\n")

    return 0 if not result.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
