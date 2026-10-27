# 本机运行

本项目使用 Python 3.11，独立环境位于 `.venv311`。
旧版 ChromaDB 的 Windows 依赖在 Python 3.13 下需要额外编译工具，建议使用 3.11。

## 配置

在 `.env` 中配置 DeepSeek 密钥（不要将密钥提交或写入本文件）：

```dotenv
OPENAI_API_KEY=填入你的密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-flash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

模型名称应以账号的 `/models` 返回结果为准。
DeepSeek 用于生成回答，本地 BGE 模型用于文档向量化；首次使用需要下载模型，后续使用本地缓存。
本机已将模型下载到 `.runtime/models/bge-small-zh-v1.5`，`.env` 的
`LOCAL_EMBEDDING_MODEL` 指向该目录的绝对路径。移动项目后需要同步修改此路径。

## 启动

在项目目录运行：

```powershell
.\.venv311\Scripts\python.exe -m app.main
```

也可以运行 `start.ps1`。网页地址为 <http://localhost:8000>，API 文档为 <http://localhost:8000/docs>。

导入示例文档：

```powershell
.\.venv311\Scripts\python.exe -m scripts.ingest data/samples
```

测试：

```powershell
.\.venv311\Scripts\python.exe -m pytest
```

## 重建环境

使用已安装的 Python 3.11 创建 `.venv311`，然后安装依赖：

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-local.txt
```

`requirements-local.lock.txt` 保存了本次为 Windows / Python 3.11 解析的运行依赖版本。

如果系统的 `python` 指向失效的 Windows Store 快捷方式，直接使用上面的虚拟环境解释器路径即可。

## 已验证

- 网页、API 文档、健康检查均返回 HTTP 200。
- 通过上传接口导入 3 份示例文档，共 8 个文本块。
- DeepSeek 基于文档回答价格问题，并返回引用来源。
- SSE 流式回答正常完成。
- 项目 33 项测试和 Ruff 静态检查通过。

本次后台服务只监听 `127.0.0.1:8000`。日志保存在 `.runtime/server.log` 和
`.runtime/server-error.log`。后续自行启动时可在终端按 Ctrl+C 停止服务。
