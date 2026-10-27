# 检索评估记录

日期：2026-10-27。Windows，Python 3.11.17，BGE-small-zh-v1.5 本地 CPU 向量化，ChromaDB，重排关闭。
数据为 data/samples 的 3 份文档（8 个文本块），查询集为 data/eval_set.example.json 的 8 道题；TOP_K=6，CHUNK_SIZE=700，CHUNK_OVERLAP=120。两种方式复用相同向量库，先混合、后纯向量，各测一次，非性能基准。

| 方式 | Hit@6 | MRR | 关键词召回率 | 平均检索耗时 |
| --- | --- | --- | --- | --- |
| 向量 + BM25 / RRF | 100.0% | 1.0000 | 89.58% | 19.06 ms |
| 纯向量 | 100.0% | 1.0000 | 89.58% | 13.74 ms |

两种方式在该示例集的质量指标相同，不能据此声称混合检索提升召回率。样本少、Top-K 接近块总数，且预期来源较宽，容易达到较高命中率。耗时受执行顺序、缓存和硬件影响；不含模型下载、首次加载或答案生成，不可推导吞吐量。
关键词召回按现有脚本对子字符串精确匹配，标签中的同义写法也分别计数，不代表语义正确率。当前结果未衡量答案忠实度或真实业务效果。

## 复现

先按主 README 配置环境和本地模型。在独立、空的知识库上导入 data/samples，确保没有其他资料，再运行：

```powershell
.\.venv311\Scripts\python.exe -m scripts.ingest data/samples
.\.venv311\Scripts\python.exe -m scripts.evaluate --dataset data/eval_set.example.json --details --json-out hybrid.json
.\.venv311\Scripts\python.exe -m scripts.evaluate --dataset data/eval_set.example.json --no-hybrid --details --json-out vector.json
```

可用环境变量 DATA_DIR、CHROMA_DIR、UPLOAD_DIR 指向隔离目录，避免修改自己的知识库。原始逐题结果见 [hybrid.json](hybrid.json) 与 [vector.json](vector.json)。

## 验证范围

离线测试覆盖分块、检索融合、入库与接口；真实模型问答另行进行本机冒烟验证。自动化测试不能代替远程模型可用性检查。

本机验证：37 项 pytest 用例通过，Ruff 检查通过；使用真实本地向量模型和已配置的 DeepSeek，完成 3 份示例文档入库、普通问答及 SSE 冒烟验证。测试隔离目录不随仓库上传。

全新环境的完整安装流程尚未验证；安装依赖和首次下载模型需要可用网络。自动化检查配置见 [GitHub Actions](../../.github/workflows/tests.yml)，执行状态以仓库 Actions 页面为准。
