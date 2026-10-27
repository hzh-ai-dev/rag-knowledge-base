"""入库管道：加载 -> 分块 -> 向量化 -> 写入向量库。

对外只需调用 ``IngestionPipeline.ingest_files()`` / ``ingest_directory()``，
上层（API 或 CLI）不关心内部细节 —— 这就是把流程编排收敛到一处的价值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.vectorstore import VectorStore
from app.ingestion.loader import iter_supported_files, load_file
from app.ingestion.splitter import split_documents
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestResult:
    """一次入库的统计结果，用于接口返回与日志。"""

    files: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "chunks": self.chunks,
            "skipped": self.skipped,
            "errors": self.errors,
        }


class IngestionPipeline:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def ingest_files(self, paths: list[str | Path]) -> IngestResult:
        """逐个文件入库。单个文件失败不影响其余文件（容错设计）。"""
        result = IngestResult()

        for path in paths:
            path = Path(path)
            try:
                documents = load_file(path)
                if not documents:
                    result.skipped.append(f"{path.name}: 无可提取文本")
                    continue

                chunks = split_documents(documents)
                if not chunks:
                    result.skipped.append(f"{path.name}: 分块后为空")
                    continue

                source = str(chunks[0].metadata["source"])
                # 幂等更新：同一文件重新入库会先清掉旧块，避免重复累积
                self._store.delete_by_source(source)
                written = self._store.add_documents(chunks)

                result.files += 1
                result.chunks += written
            except Exception as exc:  # noqa: BLE001 - 单文件失败不应中断整批
                logger.exception("入库失败: %s", path.name)
                result.errors.append(f"{path.name}: {exc}")

        return result

    def ingest_directory(self, directory: str | Path) -> IngestResult:
        files = iter_supported_files(directory)
        logger.info("扫描到 %d 个文件: %s", len(files), directory)
        return self.ingest_files(files)
