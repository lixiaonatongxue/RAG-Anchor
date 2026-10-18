# RAG-AnchorLab Backend

独立 uv 项目（Python 3.11）。当前提供健康检查接口，并包含 `pypdfium2` 技术验证测试。

## 命令

```bash
uv sync --locked
uv run python -V
uv run uvicorn rag_anchor.main:app --reload --port 8000
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

## 接口

| 方法 | 路径             | 说明                                     |
| ---- | ---------------- | ---------------------------------------- |
| GET  | `/api/v1/health` | 返回 `{"status":"ok","version":"0.1.0"}` |

## 约束

- 禁止 `pip install` / `uv pip install`，依赖一律通过 `uv add` 管理。
- 不使用系统 Python 或 CALF 环境。
- PDF 引擎优先 `pypdfium2`；PyMuPDF 在许可证确认前不安装。
