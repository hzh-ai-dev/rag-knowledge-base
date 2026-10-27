# 新手快速上手（30 分钟跑起来）

这份文档假设你**只有 Python 基础**、没做过 Web 项目。照着做,30 分钟内就能在浏览器里用上自己的知识库问答。

> 想深入了解每个技术决策背后的原因,看 `README.md`;这里只讲「怎么把它跑起来、怎么演示、面试怎么讲」。

---

## 你需要准备

1. 一台电脑(Windows 即可,下面以 Windows PowerShell 为例)
2. Python **3.10 或更高**(3.11 更顺)
3. 一个**免费**的大模型 API Key(下面带你拿)

---

## 第一步:确认 Python

打开 PowerShell,输入:

```powershell
python --version
```

显示 `Python 3.10.x` 及以上即可。没有的话去 <https://www.python.org/downloads/> 下载,**安装时务必勾选 "Add Python to PATH"**。

---

## 第二步:免费拿到智谱 GLM 的 Key(3 分钟)

本项目默认用**智谱 GLM**,因为它的对话模型 `glm-4-flash` 和向量模型 `embedding-3` **都免费**,而且一个 base_url 就能同时提供「对话」和「向量」两种能力,对新手最省事。

1. 打开 <https://open.bigmodel.cn>,注册并完成**实名认证**(免费)
2. 进入控制台 → **API Keys** → 复制你的 key(形如 `xxxxxxxx.xxxxxxxx`)

> 不想用智谱?`.env.example` 里还给了硅基流动、通义千问、DeepSeek 的配置,取消注释即可切换。

---

## 第三步:安装依赖

在 PowerShell 里进入项目目录(把路径换成你自己放项目的实际路径):

```powershell
cd 你的路径\rag-knowledge-base

# 创建独立环境(推荐,避免污染系统 Python)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

> PowerShell 若提示「禁止运行脚本 / 无法加载 Activate.ps1」,先执行一次:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`,输入 `Y` 回车,再重新激活。

---

## 第四步:填配置

```powershell
copy .env.example .env
```

用记事本打开 `.env`,把 `OPENAI_API_KEY=` 后面换成你刚才复制的 key,保存:

```env
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4-flash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=embedding-3
```

> 其余配置保持默认即可。

---

## 第五步:导入示例文档并启动

项目自带 3 份示例文档(`data/samples/`),先让系统「有东西可查」:

```powershell
python -m scripts.ingest data/samples
```

看到「入库完成:3 个文件 / N 个文本块」就成功了。然后启动服务:

```powershell
uvicorn app.main:app --reload --port 8000
```

---

## 第六步:打开浏览器用起来

访问 <http://localhost:8000>,你会看到一个聊天界面:

- **左侧**:上传你自己的文档(PDF / Word / Markdown / TXT),或删除、重建索引
- **中间**:提问,比如「私有化部署起步价是多少?」
- 回答会**逐字流式输出**,底部附上**引用来源**(哪个文件、第几页)

试试这些示例问题(答案就在 `data/samples/` 里):

- 私有化部署起步价是多少?
- 单个上传文件的最大体积限制是多少?
- 混合检索用了什么融合算法?

接口文档在 <http://localhost:8000/docs>,面试时可以展示。

---

## 常见问题

| 现象 | 原因 / 解决 |
| --- | --- |
| `ModuleNotFoundError` | 没激活虚拟环境,或依赖没装全。重新 `Activate.ps1` 后 `pip install -r requirements.txt` |
| 提问报 401/鉴权错误 | `.env` 里的 key 填错,或没保存。检查 `OPENAI_API_KEY` |
| 提问报 embedding/向量相关错误 | 你用的厂商不提供 embedding 接口(如 DeepSeek)。把 `EMBEDDING_PROVIDER=local` 并安装 `pip install sentence-transformers langchain-community` |
| 回答总是「根据已有资料无法回答」 | 知识库为空。先执行第五步入库,或检查左侧「已入库文档」是否有内容 |
| 界面能开但提问转圈很久 | 免费模型偶尔排队,耐心等一下;或在 `.env` 调小 `LLM_MAX_TOKENS`/`TOP_K` |
| 想清空重来 | 网页点「清空」,或命令行 `python -m scripts.ingest data/samples --reset` |

---

## 一周计划(把项目做扎实,面试能讲透)

**第 1 天**:按上面把项目跑起来,用示例文档提问,把网页界面和 `/docs` 都点一遍。

**第 2 天**:读 `app/core/retriever.py`(混合检索 + RRF)和 `app/core/rag_chain.py`(检索→拼上下文→生成)。这是项目的技术核心,务必读懂。

**第 3 天**:换成**你自己的文档**(课程笔记 / 某产品说明书 / 你感兴趣的开源项目文档),再提问。感受一下什么样的文档效果好、什么样的不好。

**第 4 天**:跑测试和评估:
```powershell
pytest -v
python -m scripts.evaluate --dataset data/eval_set.example.json --details
```
再跑一次对比实验,亲眼看混合检索比纯向量检索好多少:
```powershell
python -m scripts.evaluate --dataset data/eval_set.example.json --no-hybrid
```
把两次的 `Hit@6` / `MRR` 记下来——**面试时这就是你的数据**。

**第 5 天**:自己动手改一处并说明动机。最简单的加分项:调整 `CHUNK_SIZE`/`CHUNK_OVERLAP`,用评估脚本对比效果;或把 `ENABLE_RERANK` 打开看变化。

**第 6 天**:整理成简历项目 + GitHub。把 `README.md` 的「写进简历」一段按你的实际数据改写。**务必把 `.env` 加进 `.gitignore`(已默认),别把 key 传上 GitHub。**

**第 7 天**:对着 `README.md` 的「面试常见提问」自问自答,重点练这两句:
- 「为什么不用纯向量检索?」→ 专有名词/型号语义区分度低,BM25 补召回
- 「RRF 为什么好?」→ 只用排名、无需归一化两种检索的分数量纲

---

## 面试时怎么演示(3 分钟版)

1. **一句话定位**:「这是一个企业知识库 RAG 问答系统,上传文档就能基于文档内容问答,回答带引用溯源。」
2. **现场演示**:上传一份文档 → 问一个只有该文档才知道的问题 → 指出回答里的 `[1]` 引用和底部来源卡片。
3. **讲一个技术决策**:混合检索(向量 + BM25 + RRF),并给出你评估脚本里的对比数据。
4. **主动说不足**:「会话记忆目前在进程内存里,生产环境我会换成 Redis;重排目前是轻量实现,可替换成 bge-reranker。」——主动暴露已知边界,比被问出来加分。

---

## 想再进阶(可选,写进简历更亮)

按性价比排序,详见 `README.md` 的「后续可扩展方向」:

1. **查询改写**:先用大模型把口语问题改写成检索友好的表达,再检索
2. **接入真正的重排模型**:把 `HybridRetriever._lexical_rerank` 换成 `bge-reranker`
3. **父文档检索**:小块检索、大块生成,兼顾精度与完整性
4. **加权限过滤**:给文档打部门标签,检索时按用户过滤(代码里 `where` 参数已预留)
