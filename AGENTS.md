# AGENTS.md

本文件面向参与本仓库的 AI 编码助手与人类开发者，记录不可违反的工程约束。

## 项目概述

RAG-AnchorLab 是 PDF 证据定位与可视化工具。

核心流程：上传 PDF → 建立页面文本与 Chunk 索引 → 字面/语义检索 →（可选）模型基于检索证据回答 → 服务端校验引用 → `evidence/locate` 返回真实 PDF 坐标 → PDF.js 高亮。

核心后端 API（已实现）：

```python
search_anchors(pdf_path, query, *, match_case=True) -> list[AnchorHit]
```

必须正确处理：**零命中、单命中、多命中**，返回全部匹配位置。

## 目录约定

- `backend/`：**独立 uv 项目**，依赖边界是 `backend/pyproject.toml`。
- `docs/`：架构与 ADR。
- 仓库根目录**不创建** Python 项目、`.venv` 或 `pyproject.toml`。
- 前端目录 `frontend/` 已实现 PDF.js 阅读、字面/语义检索、问答与引用定位。

## Python 环境规则（强制）

1. 只用 `uv` 管理 Python 版本、虚拟环境和依赖。
2. Python 目标版本 **3.11**（`.python-version`、`requires-python` 均锁定）。
3. 使用 uv 托管的 CPython，不使用系统 Python。
4. **禁止** `pip install`，**禁止**修改 CALF/Conda 环境。
5. **禁止** `uv pip install`：所有依赖必须经 `uv add` 写入 `pyproject.toml` 与 `uv.lock`。
6. 执行命令统一用 `uv run`，例如 `uv run pytest`。
7. `uv.lock` 必须提交；不要手改锁文件。

## 技术约束

- 已实现可选能力：OpenAI 兼容服务的生成式问答（默认关闭，联调使用 Qwen3）与 Ollama `bge-m3` 语义检索（默认关闭），均通过 HTTP 调用；不引入外部向量数据库。
- 当前仍**不引入**：OCR、PyTorch、CUDA、GPU。
- 生产默认证据选择策略为 `legacy`；`neighbor_v1` 仅为实验可选策略；B2 `context_window_v1` 已归档，不属于产品功能。
- PDF 引擎优先 `pypdfium2`（已安装，M0-02 技术验证已完成）。
- PyMuPDF 在许可证确认前**不安装**。
- 不创建尚未使用的空模块或过度抽象。
- 不提交用户上传的 PDF、`.env`、API Key、代理凭据。

## 坐标契约（摘要）

详见 `docs/adr/0001-coordinate-contract.md`。

- **权威数据**：PDF 原始几何坐标（points）与 quads。
- **派生数据**：归一化坐标（0..1），仅用于前端定位，不作为事实来源。
- **渲染变换**：由 PDF.js viewport 负责，前端不做 pt→px 手工换算。
- 页码统一使用 **0-based `page_index`**，对外响应中同时给出 `page_number`（1-based）便于展示。

## 常用命令

```bash
cd backend
uv sync --locked
uv run python -V
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run uvicorn rag_anchor.main:app --reload --port 8000
```

## 变更纪律

- 修改依赖后必须提交更新后的 `uv.lock`。
- 提交前必须通过：pytest、ruff check、ruff format --check、uv lock --check。
- 不执行 `git commit` / `git push`，除非任务明确要求。
- 不修改 Git 全局配置（`user.name` / `user.email`）。
