# RAG-Anchor

**PDF 证据定位与可追溯问答**工具，前端副标题为 **PDF Evidence Explorer**。

用户上传 PDF 后，可以输入证据文字，在原文中检索并返回所有命中位置的页码与精确几何坐标；也可以针对当前文档提问，用字面或语义证据检索生成可验证引用，并可选调用 Qwen 生成带引用的回答。前端用 PDF.js 定位、高亮检索命中和引用原文。

```text
上传 PDF
   ↓
文本抽取与 Chunk 索引
   ↓
字面检索 / 可选语义检索
   ↓
（可选）Qwen 生成 → 服务端引用校验
   ↓
PDF 精确定位（页码 + PDF user space 几何坐标）
   ↓
前端 PDF.js 渲染并高亮
```

## 产品定位

RAG-Anchor 的核心不是“接收 bbox 然后画框”，而是从 PDF 原文中**检索证据**，再返回可直接用于渲染的精确行矩形（`rects`）与派生外包框（`bbox`）。

当前精确匹配 API：

```python
search_anchors(pdf_path: str | Path, query: str, *, match_case: bool = True) -> list[AnchorHit]
```

## 当前阶段

**M2：可追溯检索、精确证据定位、多语言语义检索与证据问答（默认无 LLM，可选生成式模式）**。

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
- 可选多语言语义检索：`bge-m3` embedding + SQLite 派生向量 + 余弦 Top-K；默认关闭，不引入外部向量数据库
- `evidence/locate`：把检索 span 映射为真实 PDFium 字符区间与逐行几何矩形；支持补充平面 Unicode 字符（UTF-16 代理对）
- 证据展示问答（默认）：单文档检索 → 可验证引用列表 → 点击引用复用 `evidence/locate` 与 PDF.js 高亮；不调用语言模型
- 可选 AI 生成问答：同一份检索证据 → 可配置的模型服务 → 服务端分配证据编号并要求模型返回结构化 claims → 引用字段全部由服务端重建
- 问答支持字面/语义检索与证据展示/AI 生成的正交组合；无证据时明确拒答，模型未配置或不可用时返回明确错误，不伪造成功
- Vitest 单元测试：API client、高亮坐标转换、命中居中滚动、检索引用、问答模式与错误提示、旧请求失效
- 后端 Mock Provider 测试覆盖引用校验、未知编号拒绝、非法 JSON、超时/连接失败与提示词注入隔离；自动化测试不调用真实模型
- 已使用本机 Ollama `bge-m3` + `qwen3:8b` 完成一次固定合成集回归（42 题，含生成），并用 `bge-m3` 完成首轮真实 PDF 检索探索评测（5 题，未做生成评测）；结果见 [M2 质量评测报告](docs/m2-quality-evaluation.md)，不代表生产质量

本地开发 MVP：**无认证、无速率限制、无 OCR、无外部向量数据库**。问答默认是**证据展示模式**，语义检索与生成模式默认关闭；只有显式启用并配置模型服务后才可使用。所有实测数字都来自特定模型与数据集，不代表生产质量。

## 技术栈

| 层   | 选型                                                              |
| ---- | ----------------------------------------------------------------- |
| 后端 | Python 3.11 + uv + FastAPI + Pydantic + SQLite + pytest + ruff    |
| 前端 | TypeScript 5 + Vite 5 + 原生 HTML/CSS + `pdfjs-dist` 4 + Vitest 2 |
| 解析 | `pypdfium2` + PDFium；PyMuPDF 在许可证确认前不安装                |
| 检索 | 字面 Chunk 检索；可选 `bge-m3` embedding + SQLite 派生向量 + 余弦 Top-K |
| 模型（可选） | Ollama（语义）/ OpenAI 兼容服务（生成）；默认关闭，仓库不含权重 |
| 存储 | 本地文件系统 + SQLite 元数据                                      |

前端不使用 React/Vue、UI 组件库或状态管理库。仓库内不包含模型权重、推理框架、外部向量数据库、OCR、PyTorch 或 GPU 代码；语义检索与生成式模式均通过 HTTP 调用外部模型服务，默认关闭。

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
│   ├── evaluation/          # 固定语义评测集（v2.1）
│   ├── scripts/             # 评测脚本
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
│   ├── m2-quality-evaluation.md
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
4. 点击「建立索引」，确认状态显示页数与 Chunk 数。
5. 在搜索框输入证据文字片段（例如 `quick`），选择是否区分大小写。
6. 点击「搜索」，命中结果按页码列在左侧。
7. 点击任意命中项，阅读器跳到对应页并高亮该命中，同时将命中位置滚动到阅读器视口中央（同页与跨页均支持）；其他命中使用另一种颜色。
8. 使用「上一命中 / 下一命中」在所有命中之间循环切换。
9. 使用翻页、页码输入、放大、缩小、适合宽度查看细节；缩放后命中高亮会按当前 viewport 重新计算位置。
10. 在问答面板选择检索方式：
    - **字面**：默认，直接检索 Chunk 原文。
    - **语义（需先建立语义索引）**：点击「建立语义索引」，调用已配置的 embedding 服务；索引失败时不会自动回退到字面检索。
11. 选择回答模式：
    - **证据展示（不调用语言模型）**：默认模式，始终可用。
    - **AI 生成回答（需已配置模型服务）**：仅在环境变量启用并配置模型后可用。
12. 输入问题；字面模式可另填文档中出现的检索关键词（留空则使用问题原文）；语义模式使用问题本身做 embedding。
13. 证据展示模式只返回匹配的原文证据和引用列表，并明确标注“证据展示模式，未接入语言模型”；没有证据时明确拒答。
14. AI 生成模式会用同一份证据调用模型，仅接受服务端分配的引用编号（`E1`、`E2`…）；模型输出非法或引用未知编号时返回错误，不展示该回答。模型未配置时返回 503 提示，不会伪造成功。
15. 点击任意引用（如 `E1`），前端使用 `evidence/locate` 验证 span 并跳转、高亮原始 PDF；定位失败时不会搜索文字或绘制近似高亮。
16. 悬停文档卡片右上角可删除文档（会二次确认）。

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
| `POST`   | `/api/v1/documents/{document_id}/semantic-index` | 建立或重建派生语义向量索引（可选）  |
| `POST`   | `/api/v1/retrieval/search`                       | 在已索引 Chunk 中检索并返回来源偏移 |
| `POST`   | `/api/v1/documents/{document_id}/evidence/locate` | 将检索 span 定位为真实 PDF 几何     |
| `POST`   | `/api/v1/documents/{document_id}/qa`              | 证据展示（默认）或 AI 生成问答，引用由服务端校验 |

最小后端链路：

```text
POST /api/v1/documents                      # 上传 PDF
POST /api/v1/documents/{document_id}/index  # 建立索引（可重复调用重建）
POST /api/v1/retrieval/search               # 在索引中检索
```

索引由 `DocumentIndexService` 完成：`pypdfium2` 抽取页面文本，确定性切分 Chunk，并在单个 SQLite 事务内原子替换旧索引。重复调用会重建索引，最终 Chunk 集合保持一致，不会产生重复行。检索只读取已持久化的 Chunk，返回 Python 字符串偏移，不返回 PDF 几何坐标。

搜索响应中的 `rects` 是 **absolute PDF user space**（points），前端通过 PDF.js viewport 映射到屏幕坐标。坐标语义见 [ADR-0001](docs/adr/0001-coordinate-contract.md)。

## 模型依赖（可选）

字面检索和证据展示问答不需要模型，可以完全离线运行。以下能力需要本机或远程 HTTP 服务：

| 能力 | 默认 | 依赖 | 配置 |
| --- | --- | --- | --- |
| 语义索引 / 语义检索 | 关闭 | Ollama `bge-m3`（`POST /api/embed` + `GET /api/tags`） | `RAG_ANCHOR_EMBEDDING_*` |
| AI 生成回答 | 关闭 | OpenAI 兼容 `POST /chat/completions`；本机联调用 Ollama `qwen3:8b` | `RAG_ANCHOR_LLM_*` |

模型不可用或索引过期时返回明确错误，不会回退到字面检索或伪造回答。语义索引是 SQLite 内的派生向量，不是外部向量数据库。完整配置见 [backend/README.md](backend/README.md)。

## 生成式问答配置（可选）

`grounded_generation` 默认关闭。仅配置地址不会触发任何网络请求；必须同时显式启用。

| 环境变量                        | 必填 | 默认值 | 说明                                             |
| ------------------------------- | ---- | ------ | ------------------------------------------------ |
| `RAG_ANCHOR_LLM_ENABLED`        | 是   | `false` | `1/true/yes/on` 才启用生成模式                  |
| `RAG_ANCHOR_LLM_BASE_URL`       | 是   | 空     | OpenAI 兼容服务根地址，必须为 `http(s)://...`   |
| `RAG_ANCHOR_LLM_MODEL`          | 是   | 空     | 请求体中的模型名称，不是本地文件路径             |
| `RAG_ANCHOR_LLM_API_KEY`        | 否   | 空     | 设置后以 `Authorization: Bearer ...` 发送        |
| `RAG_ANCHOR_LLM_TIMEOUT_SECONDS`| 否   | `60`   | 单次请求超时（秒）                               |
| `RAG_ANCHOR_LLM_MAX_OUTPUT_TOKENS` | 否 | `512`  | 单次生成输出 token 上限                          |
| `RAG_ANCHOR_LLM_REASONING_EFFORT` | 否 | 空   | 可选 `none/low/medium/high`；Qwen3 本地联调建议 `none`，空值不发送该字段 |

说明：

- 服务端通过 `POST {BASE_URL}/chat/completions` 调用模型，使用标准库 `urllib`，不新增依赖。
- API Key 仅从环境变量读取，不写入日志、错误响应或 Git；`.gitignore` 忽略 `models/` 与常见模型权重/密钥文件。
- 未启用、未配置或连接失败时，接口分别返回 503 / 502 / 504，不会返回固定模板伪装成生成成功。
- 检索始终限定在当前文档；字面模式不做 embedding/向量检索，语义模式使用 `bge-m3` 派生向量；不做 rerank、查询改写、多轮对话或跨文档检索。
- 服务端给检索证据分配 `E1`、`E2`…；模型返回 `answerable` + `claims`（每条含 `text` 与 `citation_ids`）+ `limitations`，页码、偏移、Chunk ID、文档 ID 和 PDF 几何全部由服务端重建。结构校验不证明事实蕴含，生成的结论仍需人工核对原文。
- 已在本机 Ollama `qwen3:8b` 的 OpenAI 兼容端点完成固定合成集回归；真实 PDF 探索评测只做检索与原文检查，未调用生成模型。Qwen3 默认 thinking 会占用输出 token，本地联调设置 `RAG_ANCHOR_LLM_REASONING_EFFORT=none`。自动化测试仍只使用 Fake/Mock Provider。

示例（仅在模型服务已部署且你确认可调用后使用）：

```bash
cd backend
export RAG_ANCHOR_LLM_ENABLED=true
export RAG_ANCHOR_LLM_BASE_URL=http://127.0.0.1:11434/v1
export RAG_ANCHOR_LLM_MODEL=<模型名称>
# Qwen3 等 thinking 模型可关闭推理输出：export RAG_ANCHOR_LLM_REASONING_EFFORT=none
# 如服务需要鉴权：export RAG_ANCHOR_LLM_API_KEY=...
uv run uvicorn rag_anchor.main:app --port 8000
```

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

后端测试使用注入的 Fake/Mock Provider，不发起真实模型请求，也不需要 GPU 或模型文件。覆盖默认 `evidence_only`、未配置 503、引用编号校验、未知编号拒绝、非法 JSON、超时/连接失败、无证据不调用模型、提示词注入隔离与跨文档拒绝。

M2 的质量结果、B1/B2 实验与已知限制见 [M2 质量评测报告](docs/m2-quality-evaluation.md)；固定评测集位于 `backend/evaluation/semantic_dataset.json`。真实模型评测需要显式配置服务，`--provider fake` 只验证工程管线。

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
- 根目录 `test_data/` 是本地真实 PDF 语料，已在 `.gitignore` 中精确忽略；不要提交、移动或删除其中的文件。
- 前端仅通过 Vite 开发代理访问后端，不添加 CORS。

## 已知限制

- 本地开发 MVP：无认证、无速率限制、无审计。
- 不做 OCR：扫描件、图片型 PDF 无可检索文本。
- 单页渲染（只渲染当前页），不做连续滚动或虚拟化；命中自动居中只滚动阅读器视口。
- PDF 通过 `fetch` 全量加载到内存（后端上传上限 20 MiB），不依赖 HTTP Range。
- 不支持加密 PDF、任意角度旋转文字的精确四边形、RTL、复杂表格阅读顺序。
- 锚点搜索每次重新打开 PDF，无坐标缓存；Chunk 检索只读取已建立索引的数据。
- SQLite 短连接，适合单机开发，不适合多 worker 高并发写入。
- 语义检索只支持单文档，使用 SQLite 派生向量上的暴力余弦 Top-K；0.30 是当前配置，不是已验证的最优阈值。
- 复杂多证据召回仍不完整；已进入 Top-5 的证据仍可能被生成模型遗漏；模型也可能过度回答。
- 生成式问答只校验“引用了哪些服务端证据”，不验证生成句子的每个事实；引用可定位/结构合法不等于事实蕴含，答案仍可能出错，必须人工核对。
- 中文专利专项验证尚未完成；首轮 5 道真实 PDF 探索题与其余 7 道真实问题标签仍为 Draft，不是正式 benchmark。
- M2 数字来自一次固定合成集回归与首轮真实 PDF 探索评测，不能当作生产准确率；真实浏览器已验证 PDF 阅读与文档切换、Chunk 字面检索、语义索引、中文问题查询英文证据、引用跳转与高亮、拒答行为及旧 QA 高亮清理（修复后的 5 项复验全部通过），但快速连续提问、模型故障、语义索引异常/过期与完整 A–G 边界场景尚未全部人工验收。
- 不包含 Agent、reranker、专利专项检索、事实蕴含验证或跨文档检索。
- 已用本机 Ollama `bge-m3` + `qwen3:8b` 完成固定合成集回归，并用 `bge-m3` 完成首轮真实 PDF 检索探索评测，但未覆盖其他模型、较大规模请求或复杂文档；不能据此宣称泛化生成质量，生产环境仍需人工核对。

## 文档

- [架构说明](docs/architecture.md)
- [M2 质量评测报告](docs/m2-quality-evaluation.md)
- [ADR-0001：坐标契约](docs/adr/0001-coordinate-contract.md)
- [M0-06C 锚点评测可复现性审查](docs/benchmark-audit.md)
- [人工浏览器验收清单](docs/manual-ui-acceptance.md)
- [前端说明](frontend/README.md)
- [后端说明](backend/README.md)

## License

待确认，当前仓库**不包含** LICENSE 文件。
