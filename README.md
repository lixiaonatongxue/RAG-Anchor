# RAG-AnchorLab

开源的 **PDF 证据定位与可视化**工具。

用户上传 PDF 并输入一段证据文字，后端在 PDF 原文中检索该文字，返回所有命中位置的页码与精确几何坐标，前端在 PDF 上定位并高亮。

## 产品定位（重要）

RAG-AnchorLab 的核心不是“接收 bbox 然后画框”，而是：

```text
上传 PDF + 输入证据文字
        ↓
后端检索 PDF 原文（search_anchors）
        ↓
返回 页码 + quads/bbox 几何坐标（支持零命中 / 单命中 / 多命中）
        ↓
前端用 PDF.js 渲染并高亮全部命中位置
```

计划的核 Python API：

```python
search_anchors(pdf_path: str, query: str, match_mode: str = "exact") -> list[AnchorHit]
```

## 当前阶段

**M0-02：PDF 引擎技术验证**。当前包含：

- 独立 uv 后端项目
- 最小 FastAPI 应用与 `GET /api/v1/health`
- `pypdfium2` 文本搜索与页面几何 spike 测试（PDF 动态生成）
- 项目文档与坐标契约 ADR（区分已验证与未确定行为）

尚未实现生产版 PDF 检索 API。

## 技术栈

| 层   | 选型                                                  |
| ---- | ----------------------------------------------------- |
| 后端 | Python 3.11 + uv + FastAPI + Pydantic + pytest + ruff |
| 前端 | React + TypeScript + Vite + PDF.js（后续阶段）        |
| 解析 | `pypdfium2` + PDFium；PyMuPDF 在许可证确认前不安装    |

第一版不接入 LLM、向量数据库、OCR、PyTorch 或 GPU。

## 目录结构

```text
RAG-Anchor/
├── backend/                 # 独立 uv 项目
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── src/rag_anchor/
│   └── tests/
├── docs/
│   ├── architecture.md
│   └── adr/0001-coordinate-contract.md
├── README.md
├── AGENTS.md
└── .gitignore
```

## 快速开始（后端）

所有命令都在 `backend/` 下执行，且必须使用 `uv`：

```bash
cd backend

uv sync --locked          # 按 uv.lock 创建 .venv
uv run python -V          # 期望 Python 3.11.x
uv run uvicorn rag_anchor.main:app --reload --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/v1/health
# {"status":"ok","version":"0.1.0"}
```

质量校验：

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

## 环境约定

- Python 解释器使用 **uv 托管的 CPython 3.11**，不使用系统 Python，不修改 CALF/Conda 环境。
- 禁止 `pip install`，禁止 `uv pip install` 绕过锁文件。
- `.venv/`、缓存、用户上传的 PDF 不提交 Git。

## 文档

- [架构说明](docs/architecture.md)
- [ADR-0001：坐标契约](docs/adr/0001-coordinate-contract.md)

## License

待确认，当前仓库**不包含** LICENSE 文件。
