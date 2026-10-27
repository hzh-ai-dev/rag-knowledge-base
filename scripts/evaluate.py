"""检索质量评估脚本。

为什么要做评估？
RAG 项目的常见误区是「感觉回答还行」就上线。分块大小、是否混合检索、top_k 取多少，
这些决策如果没有量化指标，就只能靠拍脑袋。本脚本提供三个不需要人工标注的离线指标：

- Hit@k     ：top-k 结果里是否命中了预期来源文件（衡量「找没找到」）
- MRR       ：第一个正确结果排名的倒数均值（衡量「排得准不准」）
- 关键词召回 ：检索到的上下文里覆盖了多少预期关键词（衡量「内容对不对」）

用法：
    python -m scripts.evaluate --dataset data/eval_set.example.json --top-k 6
    python -m scripts.evaluate --dataset data/eval_set.example.json --no-hybrid   # 对比实验

数据格式（JSON 数组）：
    [
      {"question": "问题", "expected_sources": ["来源文件名.md"], "expected_keywords": ["关键词"]}
    ]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.container import get_retriever  # noqa: E402
from app.utils.logging import setup_logging  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("数据集必须是 JSON 数组")
    return data


def evaluate(dataset: list[dict], top_k: int) -> dict:
    retriever = get_retriever()
    hits = 0
    reciprocal_ranks: list[float] = []
    keyword_recalls: list[float] = []
    latencies: list[float] = []
    details: list[dict] = []

    for item in dataset:
        question = item["question"]
        expected_sources = set(item.get("expected_sources", []))
        expected_keywords = item.get("expected_keywords", [])

        start = time.perf_counter()
        chunks = retriever.retrieve(question, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)

        retrieved_sources = [c.source for c in chunks]
        context = "\n".join(c.content for c in chunks)

        # --- Hit@k 与 MRR ---
        rank = 0
        for index, source in enumerate(retrieved_sources, start=1):
            if source in expected_sources:
                rank = index
                break
        hit = rank > 0
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        # --- 关键词召回 ---
        if expected_keywords:
            found = sum(1 for kw in expected_keywords if kw in context)
            keyword_recalls.append(found / len(expected_keywords))

        details.append(
            {
                "question": question,
                "hit": hit,
                "rank": rank or None,
                "retrieved": retrieved_sources,
                "expected": sorted(expected_sources),
            }
        )

    total = len(dataset) or 1
    return {
        "questions": len(dataset),
        "top_k": top_k,
        "hit_at_k": hits / total,
        "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "keyword_recall": statistics.fmean(keyword_recalls) if keyword_recalls else 0.0,
        "avg_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "details": details,
    }


def print_report(metrics: dict, show_details: bool) -> None:
    print("\n" + "=" * 60)
    print("  检索评估报告")
    print("=" * 60)
    print(f"  问题数量         : {metrics['questions']}")
    print(f"  Top-K            : {metrics['top_k']}")
    print(f"  Hit@{metrics['top_k']}           : {metrics['hit_at_k']:.1%}")
    print(f"  MRR              : {metrics['mrr']:.4f}")
    print(f"  关键词召回率      : {metrics['keyword_recall']:.1%}")
    print(f"  平均检索耗时      : {metrics['avg_latency_ms']:.1f} ms")
    print("=" * 60)

    if show_details:
        print("\n逐条结果：")
        for index, row in enumerate(metrics["details"], start=1):
            mark = "✓" if row["hit"] else "✗"
            rank = f"rank={row['rank']}" if row["rank"] else "未命中"
            print(f"  {mark} [{index}] {row['question']}")
            print(f"      期望: {row['expected']}")
            print(f"      实际: {row['retrieved'][:3]}  ({rank})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量评估")
    parser.add_argument("--dataset", required=True, help="评估数据集 JSON 路径")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="检索返回条数")
    parser.add_argument("--no-hybrid", action="store_true", help="关闭混合检索（用于对比）")
    parser.add_argument("--details", action="store_true", help="打印逐条结果")
    parser.add_argument("--json-out", help="将指标写入指定 JSON 文件")
    args = parser.parse_args()

    setup_logging(False)

    if args.no_hybrid:
        settings.enable_hybrid = False
        print("[对比模式] 已关闭混合检索，仅使用向量检索")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"数据集不存在: {dataset_path}")
        return 1

    dataset = load_dataset(dataset_path)
    metrics = evaluate(dataset, args.top_k)
    print_report(metrics, args.details)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"指标已写入 {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
