# RAG-Anchor

开源的 **PDF 证据定位与可视化**工具，前端副标题为 **PDF Evidence Explorer**。

用户上传 PDF 并输入一段证据文字，后端在 PDF 原文中检索该文字，返回所有命中位置的页码与精确几何坐标，前端用 PDF.js 定位并高亮全部命中位置。

```text
上传 PDF
   ↓
输入证据文字 query
   ↓
后端检索原文（search_anchors）
   ↓
返回 页码 + PDF user space 几何坐标（支持零命中 / 单命中 / 多命中）
   ↓
前端 PDF.js 渲染并高亮全部命中位置
```

## 产品定位

RAG-Anchor 的核心不是“接收 bbox 然后画框”，而是从 PDF 原文中**检索证据**，再返回可直接用于渲染的精确行矩形（`rects`）与派生外包框（`bbox`）。

当前精确匹配 API：

```python
search_anchors(pdf_path: str | Path, query: str, *, match_case: bool = True) -> list[AnchorHit]
```

## 当前阶段

**M1：可追溯检索、精确证据定位与证据展示问答（无 LLM）**。

已实现：

- 独立 uv 后端项目（Python 3.11）
- FastAPI 文档上传、列表、详情、删除、文件读取与锚点搜索 API
- `pypdfium2` 精确文本检索，返回 absolute PDF user-space 坐标（ADR-0001）
- 原生 HTML/CSS/TypeScript 前端（Vite + `pdfjs-dist`）
- PDF 上传（选择 + 拖放）、文档列表与删除
- PDF.js 单页阅读器：翻页、页码跳转、放大、缩小、适合宽度
- 证据搜索、命中列表、上一命中/下一命中、高亮定位
- 点击任意命中后自动定位并居中；同页与跨页均支持，缩放后高亮位置按当前 viewport 重新计算
- Chunk 索引、字面 Chunk 检索、Python 字符串偏移与来源一致性校验
- `evidence/locate`：把检索 span 映射为真实 PDFium 字符区间与逐行几何矩形
- 证据展示问答：单文档字面检索 → 可验证引用列表 → 点击引用复用 `evidence/locate` 与 PDF.js 高亮
- 无相关证据时明确拒答；不调用语言模型，不生成无引用的事实
- Vitest 单元测试：API client、高亮坐标转换、命中居中滚动、检索引用与问答交互

本地开发 MVP：**无认证、无速率限制、无 OCR、无 LLM/向量数据库**。问答为**证据展示模式**，不是生成式 RAG。

## 技术栈

| 层   | 选型                                                              |
| ---- | ----------------------------------------------------------------- |
| 后端 | Python 3.11 + uv + FastAPI + Pydantic + SQLite + pytest + ruff    |
| 前端 | TypeScript 5 + Vite 5 + 原生 HTML/CSS + `pdfjs-dist` 4 + Vitest 2 |
| 解析 | `pypdfium2` + PDFium；PyMuPDF 在许可证确认前不安装                |
| 存储 | 本地文件系统 + SQLite 元数据                                      |

前端不使用 React/Vue、UI 组件库或状态管理库。第一版不接入 LLM、向量数据库、OCR、PyTorch 或 GPU。

## 目录结构

```text
RAG-Anchor/
├── backend/                 # 独立 uv 项目
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── .python-version
│   ├── src/rag_anchor/
│   │   ├── api/             # HTTP 路由、Pydantic 模型
│   │   ├── services/        # 锚点搜索、文档管理
│   │   ├── infra/           # SQLite 仓储
│   │   ├── domain/          # 框架无关领域模型
│   │   ├── config.py
│   │   └── main.py
│   └── tests/
├── frontend/                # Vite + TypeScript 前端
│   ├── index.html
│   ├── package.json
│   ├── src/
│   │   ├── main.ts          # UI 控制器
│   │   ├── styles.css
│   │   ├── api/             # API client 与类型
│   │   ├── pdf/             # PDF.js viewer
│   │   └── highlight/       # 坐标转换与高亮 overlay
│   └── tests/
├── docs/
│   ├── architecture.md
│   └── adr/0001-coordinate-contract.md
├── README.md
├── AGENTS.md
└── .gitignore
```

## 快速开始

### 1. 启动后端

所有 Python 命令都在 `backend/` 下执行，且必须使用 `uv`：

```bash
cd backend
export UV_LINK_MODE=copy

uv sync --locked          # 按 uv.lock 创建 .venv
uv run python -V          # 期望 Python 3.11.x
uv run uvicorn rag_anchor.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/v1/health
# {"status":"ok","version":"0.1.0"}
```

### 2. 启动前端

需要 Node.js **18.20.8 或更高**（`engines.node: >=18.0.0`）与 npm 10.x。

```bash
cd frontend
npm install
npm run dev
```

开发服务器监听 `0.0.0.0:5173`，浏览器访问 `http://<host>:5173`。

Vite 会把 `/api` 代理到 `http://127.0.0.1:8000`，因此后端不需要开启 CORS。
如需修改代理目标：

```bash
cp .env.example .env.local
# 编辑 .env.local
VITE_API_PROXY_TARGET=http://127.0.0.1:9000
```

修改后需重启开发服务器。详见 [frontend/README.md](frontend/README.md)。

## 演示步骤

1. 启动后端与前端。
2. 在左侧上传一个文本型 PDF（拖放或点击选择）。
3. 上传成功后新文档会被自动选中，右侧显示 PDF 第一页。
4. 在搜索框输入证据文字片段（例如 `quick`），选择是否区分大小写。
5. 点击「搜索」，命中结果按页码列在左侧。
6. 点击任意命中项，阅读器跳到对应页并高亮该命中，同时将命中位置滚动到阅读器视口中央（同页与跨页均支持）；其他命中使用另一种颜色。
7. 使用「上一命中 / 下一命中」在所有命中之间循环切换。
8. 使用翻页、页码输入、放大、缩小、适合宽度查看细节；缩放后命中高亮会按当前 viewport 重新计算位置。
9. 在「证据问答」中输入问题；由于当前是字面检索，可另填文档中出现的检索关键词（留空则使用问题原文）。
10. 点击「提问」后，系统只返回匹配的原文证据和引用列表，并明确标注“证据展示模式，未接入语言模型”；没有证据时会明确拒答。
11. 点击任意引用（如 `E1`），前端使用 `evidence/locate` 验证 span 并跳转、高亮原始 PDF；定位失败时不会搜索文字或绘制近似高亮。
12. 悬停文档卡片右上角可删除文档（会二次确认）。

## API 概览

| 方法     | 路径                                             | 说明                                |
| -------- | ------------------------------------------------ | ----------------------------------- |
| `GET`    | `/api/v1/health`                                 | 健康检查                            |
| `POST`   | `/api/v1/documents`                              | 上传 PDF（multipart 字段名 `file`） |
| `GET`    | `/api/v1/documents`                              | 分页列出文档                        |
| `GET`    | `/api/v1/documents/{document_id}`                | 文档详情                            |
| `DELETE` | `/api/v1/documents/{document_id}`                | 删除文档                            |
| `GET`    | `/api/v1/documents/{document_id}/file`           | 读取 PDF 原始字节                   |
| `POST`   | `/api/v1/documents/{document_id}/anchors/search` | 精确检索证据文字                    |
| `POST`   | `/api/v1/documents/{document_id}/index`          | 建立或重建 Chunk 索引               |
| `POST`   | `/api/v1/retrieval/search`                       | 在已索引 Chunk 中检索并返回来源偏移 |
| `POST`   | `/api/v1/documents/{document_id}/evidence/locate` | 将检索 span 定位为真实 PDF 几何     |
| `POST`   | `/api/v1/documents/{document_id}/qa`              | 证据展示问答，返回可验证引用（无 LLM） |

最小后端链路：

```text
POST /api/v1/documents                      # 上传 PDF
POST /api/v1/documents/{document_id}/index  # 建立索引（可重复调用重建）
POST /api/v1/retrieval/search               # 在索引中检索
```

索引由 `DocumentIndexService` 完成：`pypdfium2` 抽取页面文本，确定性切分 Chunk，并在单个 SQLite 事务内原子替换旧索引。重复调用会重建索引，最终 Chunk 集合保持一致，不会产生重复行。检索只读取已持久化的 Chunk，返回 Python 字符串偏移，不返回 PDF 几何坐标。

搜索响应中的 `rects` 是 **absolute PDF user space**（points），前端通过 PDF.js viewport 映射到屏幕坐标。坐标语义见 [ADR-0001](docs/adr/0001-coordinate-contract.md)。

## 质量校验

后端：

```bash
cd backend
export UV_LINK_MODE=copy
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

前端：

```bash
cd frontend
npm run typecheck
npm test
npm run build
```

## 环境约定

- Python 解释器使用 **uv 托管的 CPython 3.11**，不使用系统 Python，不修改 CALF/Conda 环境。
- 禁止 `pip install`，禁止绕过锁文件安装依赖。
- `.venv/`、`node_modules/`、构建产物、缓存、用户上传的 PDF 与 SQLite 数据库不提交 Git。
- 前端仅通过 Vite 开发代理访问后端，不添加 CORS。

## 已知限制

- 本地开发 MVP：无认证、无速率限制、无审计。
- 不做 OCR：扫描件、图片型 PDF 无可检索文本。
- 单页渲染（只渲染当前页），不做连续滚动或虚拟化；命中自动居中只滚动阅读器视口。
- PDF 通过 `fetch` 全量加载到内存（后端上传上限 20 MiB），不依赖 HTTP Range。
- 不支持加密 PDF、任意角度旋转文字的精确四边形、RTL、复杂表格阅读顺序。
- 锚点搜索每次重新打开 PDF，无坐标缓存；Chunk 检索只读取已建立索引的数据。
- SQLite 短连接，适合单机开发，不适合多 worker 高并发写入。

## 文档

- [架构说明](docs/architecture.md)
- [ADR-0001：坐标契约](docs/adr/0001-coordinate-contract.md)
- [前端说明](frontend/README.md)
- [后端说明](backend/README.md)

## License

待确认，当前仓库**不包含** LICENSE 文件。
