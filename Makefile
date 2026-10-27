# 常用开发命令（Linux / macOS / Git Bash；Windows 用户可参考 README 中的 PowerShell 等价命令）
.DEFAULT_GOAL := help

.PHONY: help install dev test lint format ingest eval docker-build docker-up docker-down clean

help:  ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## 安装运行依赖
	pip install -r requirements.txt

dev:  ## 启动开发服务（热重载）
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:  ## 运行测试
	pytest -v

lint:  ## 静态检查
	ruff check app tests scripts

format:  ## 自动修复代码风格
	ruff check --fix app tests scripts

ingest:  ## 导入示例文档（先清空知识库）
	python -m scripts.ingest data/samples --reset

eval:  ## 运行检索评估
	python -m scripts.evaluate --dataset data/eval_set.example.json --details

docker-build:  ## 构建镜像
	docker build -t rag-knowledge-base:latest .

docker-up:  ## 启动容器
	docker compose up -d --build

docker-down:  ## 停止容器
	docker compose down

clean:  ## 清理运行期数据与缓存
	rm -rf data/chroma data/uploads .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
