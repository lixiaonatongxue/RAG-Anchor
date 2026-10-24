# RAG-AnchorLab Backend

独立 uv 项目（Python 3.11），FastAPI 后端。当前提供健康检查、PDF 文档管理、证据搜索、索引与字面 Chunk 检索、精确证据定位、可选的多语言语义检索、证据展示/生成式问答与 PDF 文件读取接口。生成式问答与语义检索默认关闭，只有显式配置模型服务后才可用。

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

运行时数据默认写入 `backend/.data/`，可用 `RAG_ANCHOR_DATA_DIR` 覆盖：

```text
.data/
├── documents/        # {document_id}.pdf
├── tmp/              # 上传中的临时文件
└── metadata.sqlite3  # 文档元数据
```

上传大小默认上限 20 MiB，可用 `RAG_ANCHOR_MAX_UPLOAD_BYTES` 覆盖。

生成式问答（可选）环境变量：

| 变量                                 | 默认值  | 说明                                           |
| ------------------------------------ | ------- | ---------------------------------------------- |
| `RAG_ANCHOR_LLM_ENABLED`             | `false` | 必须为 `1/true/yes/on` 才启用                  |
| `RAG_ANCHOR_LLM_BASE_URL`            | 空      | OpenAI 兼容服务根地址（`http(s)://`）          |
| `RAG_ANCHOR_LLM_MODEL`               | 空      | 模型名称，不是本地文件路径                     |
| `RAG_ANCHOR_LLM_API_KEY`             | 空      | 可选；只从环境变量读取，不写入日志或响应       |
| `RAG_ANCHOR_LLM_TIMEOUT_SECONDS`     | `60`    | 单次请求超时                                   |
| `RAG_ANCHOR_LLM_MAX_OUTPUT_TOKENS`   | `512`   | 输出 token 上限                                |
| `RAG_ANCHOR_LLM_REASONING_EFFORT`    | 空      | 可选 `none/low/medium/high`；Qwen3 联调建议 `none` |

未启用或缺少地址/模型时，生成模式返回 503；Provider 构造阶段不探测网络，也不会在导入时发起请求。

Qwen3 等 thinking 模型默认可能先输出推理内容，在较小的 `max_tokens` 下导致 `content` 为空。Ollama 的 OpenAI 兼容端点可通过请求字段 `reasoning_effort: "none"` 关闭 thinking；本服务仅在显式设置 `RAG_ANCHOR_LLM_REASONING_EFFORT=none` 时发送该字段，未设置时保持原请求格式。

语义检索（M2-01，可选）环境变量：

| 变量                                  | 默认值 | 说明                                                        |
| ------------------------------------- | ------ | ----------------------------------------------------------- |
| `RAG_ANCHOR_EMBEDDING_ENABLED`        | `false` | 必须为 `1/true/yes/on` 才启用                               |
| `RAG_ANCHOR_EMBEDDING_BASE_URL`       | 空      | Ollama 根地址，例如 `http://127.0.0.1:11434`                |
| `RAG_ANCHOR_EMBEDDING_MODEL`          | 空      | embedding 模型名，例如 `bge-m3`；不得使用聊天模型            |
| `RAG_ANCHOR_EMBEDDING_MODEL_VERSION`  | 空      | 运维修订标签，**不是模型身份证明**；空值记录为 `unversioned` |
| `RAG_ANCHOR_EMBEDDING_API_KEY`        | 空      | 可选；只从环境变量读取，不写入日志或响应                     |
| `RAG_ANCHOR_EMBEDDING_TIMEOUT_SECONDS`| `120`   | 单次 embed 请求超时                                          |
| `RAG_ANCHOR_EMBEDDING_BATCH_SIZE`     | `16`    | 建索引时每批文本数                                           |
| `RAG_ANCHOR_EMBEDDING_MIN_SCORE`      | `0.30`  | 语义候选的最低余弦相似度                                     |

Provider 使用 Ollama 原生 `POST /api/embed`，仅在显式建立语义索引或提交语义问答时发起请求。建立或查询语义索引前，Provider 会通过 `GET /api/tags` 找到与 `RAG_ANCHOR_EMBEDDING_MODEL` 匹配的模型，读取其 digest，并与模型名一起组成稳定身份（`name + digest`）写入派生索引、在查询前重新校验。

- 模型名不存在、条目缺少 digest、或多个同名条目 digest 冲突时，视为身份无法确认，返回 502，不会把不确定性当作“未变化”。
- `/api/tags` 请求超时返回 504；HTTP 错误或传输失败返回 502。
- 每次显式建立/查询只发起一次身份请求，不做跨请求长期缓存；查询前若 digest 与索引记录不一致，返回 409（模型已变化，需重建索引）。
- `RAG_ANCHOR_EMBEDDING_MODEL_VERSION` 只是运维修订标签，用于人工换代提示；它不参与模型身份验证，也不能替代 digest 校验。

本机尚无 `bge-m3` 时请先执行 `ollama pull bge-m3`（需人工授权），不要用 `qwen3:8b` 等聊天模型代替 embedding 模型。

## 接口

| 方法   | 路径                                             | 说明                                        |
| ------ | ------------------------------------------------ | ------------------------------------------- |
| GET    | `/api/v1/health`                                 | 返回 `{"status":"ok","version":"0.1.0"}`    |
| POST   | `/api/v1/documents`                              | 上传 PDF（multipart 字段 `file`）           |
| GET    | `/api/v1/documents`                              | 分页列出文档（`limit` 1–100，`offset` ≥ 0） |
| GET    | `/api/v1/documents/{document_id}`                | 查询单个文档元数据                          |
| DELETE | `/api/v1/documents/{document_id}`                | 删除文档元数据与存储文件                    |
| POST   | `/api/v1/documents/{document_id}/anchors/search` | 检索证据文字并返回全部命中位置              |
| POST   | `/api/v1/documents/{document_id}/index`          | 建立或重建可追溯的页面文本与 Chunk 索引     |
| POST   | `/api/v1/documents/{document_id}/semantic-index` | 建立或重建派生语义向量索引（不改动原文索引） |
| POST   | `/api/v1/retrieval/search`                       | 在已索引 Chunk 中检索并返回来源偏移         |
| POST   | `/api/v1/documents/{document_id}/evidence/locate` | 将检索 span 定位为真实 PDF 几何            |
| POST   | `/api/v1/documents/{document_id}/qa`             | 证据展示（默认）或 AI 生成问答，引用由服务端校验 |
| GET    | `/api/v1/documents/{document_id}/file`           | 读取已上传 PDF 原始字节（inline）           |

### 证据搜索

请求：

```http
POST /api/v1/documents/{document_id}/anchors/search
Content-Type: application/json

{
  "query": "The proposed method",
  "match_case": true
}
```

响应（所有命中，按 `(page_index, char_start)` 排序）：

```json
{
  "document_id": "6f1c9e2a-1b2c-4d5e-8f90-123456789abc",
  "query": "The proposed method",
  "match_case": true,
  "total": 1,
  "results": [
    {
      "page_index": 0,
      "page_number": 1,
      "matched_text": "The proposed method",
      "char_start": 42,
      "char_count": 19,
      "rects": [
        { "left": 100.168, "bottom": 200.0, "right": 240.436, "top": 208.616 }
      ],
      "bbox": {
        "left": 100.168,
        "bottom": 200.0,
        "right": 240.436,
        "top": 208.616
      },
      "page": {
        "page_index": 0,
        "width_pt": 612.0,
        "height_pt": 792.0,
        "rotation": 0,
        "cropbox": [0.0, 0.0, 612.0, 792.0]
      }
    }
  ]
}
```

- `rects`：每个匹配文本行一个矩形；跨行命中保留多行矩形，不返回跨行大框。
- `bbox`：所有行矩形的派生外包框，不能替代 `rects`。
- 零命中返回 HTTP 200，`total` 为 `0`，`results` 为 `[]`。

### 文档索引

```http
POST /api/v1/documents/{document_id}/index
```

无请求体。第一版始终使用服务默认切分参数（`max_chars=800`、`overlap_chars=0`）。

响应（HTTP 200）：

```json
{
  "document_id": "6f1c9e2a-1b2c-4d5e-8f90-123456789abc",
  "fingerprint": "0f1c...9a4d",
  "page_count": 3,
  "chunk_count": 3,
  "filtered_chunk_count": 0,
  "indexed_at": "2026-10-22T07:41:00.123456Z"
}
```

- 该接口只调用 `DocumentIndexService.index_document()`；PDF 解析、切分、指纹计算与旧索引清理都不在路由层重写。
- 每次调用都会重新解析 PDF 并重建索引。相同 PDF 与相同配置下最终 Chunk 集合一致，不会出现重复行；这不是缓存命中。
- 索引在单个 SQLite 事务内原子替换：解析、切分或写入失败时，原有索引保持不变。
- 索引只是“已建立”（`indexed`），不代表已经校验过与当前 PDF 一致（`current`）；本版本不提供索引有效性检查接口。
- 文档删除时，`document_page_texts`、`document_chunks`、`document_index_metadata` 由外键级联清理。

### Chunk 检索

```http
POST /api/v1/retrieval/search
Content-Type: application/json

{
  "query": "The proposed method",
  "document_id": "6f1c9e2a-1b2c-4d5e-8f90-123456789abc",
  "match_case": true,
  "limit": 20,
  "offset": 0
}
```

- `document_id` 可省略；省略时检索所有已索引文档。
- 响应返回 `total`、分页参数与命中列表。每个命中包含 `chunk_id`、`page_index`、`page_number`、`chunk_index`、`start_offset`、`end_offset`、`match_start`、`match_end`、`hit_text`、`chunk_text`、`extraction_version`。
- `start_offset` / `end_offset` 是 Chunk 在页面文本中的 Python 字符串偏移；`match_start` / `match_end` 是命中在同一页面文本中的偏移。它们都不是 PDF 坐标。
- 检索只读已持久化的 Chunk，不解析 PDF、不建立索引、不写数据库。

### 语义索引与语义检索

```http
POST /api/v1/documents/{document_id}/semantic-index
```

无请求体。前置条件：文档已建立字面索引，且 embedding 服务已启用并配置。

响应（HTTP 200）：

```json
{
  "document_id": "6f1c9e2a-1b2c-4d5e-8f90-123456789abc",
  "model_name": "bge-m3",
  "model_version": "unversioned",
  "model_digest": "sha256:5f1c...9a4d",
  "dimension": 1024,
  "normalization": "l2",
  "index_version": 1,
  "candidate_version": 1,
  "source_fingerprint": "0f1c...9a4d",
  "candidate_count": 42,
  "built_at": "2026-10-24T08:00:00Z"
}
```

- 语义索引是**派生的向量索引**，存于同一 SQLite 的 `semantic_index_metadata` / `semantic_candidates` 表，不替代 `document_page_texts`、`document_chunks` 或原文 PDF。
- `model_version` 是运维标签，`model_digest` 是真正的模型身份证据；`index_version` / `candidate_version` 分别标识向量索引格式与候选切分规则版本。
- 候选来自存储的页面文本与 Chunk，按句子/受控窗口切分；每条候选保留 `document_id`、`chunk_id`、`page_index`、`page_number`、`chunk_index`、`start_offset`、`end_offset` 与原文。建索引时逐条校验 `page_text[start:end] == source_text`、完全位于 Chunk 内，且 Chunk/页码/序号与索引记录一致。
- 向量写入前做 L2 归一化，检索使用余弦相似度（归一化后的点积）与 Top-K；排序并列时按页码、Chunk、偏移和候选 ID 稳定排序。
- 读取向量时严格校验 BLOB：长度必须是 4 的倍数、维度与元数据一致、所有分量为有限浮点、范数非零且接近 1（float32 容差）。非法向量使整次检索返回 409，不会参与排序或产生部分结果。
- 嵌入只决定相关性排序，**不会生成或修正字符偏移**；无法映射回原文的候选会被拒绝，而不是伪造高亮。
- 失效规则：重新建立字面索引会删除旧语义索引；模型名/版本、模型 digest、索引格式版本、候选切分版本、归一化规则、源指纹、提取版本、候选数量或向量维度不一致时，问答返回 409，不会静默使用旧向量。
- 旧数据库通过 `ALTER TABLE ... ADD COLUMN` 增量迁移到新元数据列，默认值为空 digest 和版本 0；缺少版本信息的旧索引一律视为过期，不会删除原文或派生数据。重建失败时旧语义索引与原文保持不变。
- 错误语义：文档不存在优先返回 404（即使 embedding 未配置）；建索引过程中文档被删除返回 404；409 未建立字面索引、版本/digest/指纹不一致或语义索引为空；503 embedding 未启用/未配置；504 embedding 或 `/api/tags` 超时；502 embedding 服务失败、模型身份无法确认或返回非法向量；500 索引持久化失败。

### 问答

```http
POST /api/v1/documents/{document_id}/qa
Content-Type: application/json

{
  "question": "论文提出的方法有哪些特点？",
  "retrieval_query": "proposed method",
  "match_case": false,
  "max_citations": 5,
  "answer_mode": "evidence_only",
  "retrieval_mode": "literal"
}
```

- `answer_mode` 可选：`evidence_only`（默认，不调用语言模型）或 `grounded_generation`（可选，需配置模型服务）。
- `retrieval_mode` 可选：`literal`（默认，字面 Chunk 检索）或 `semantic`（多语言语义检索，需先建立语义索引）。
- `literal` 模式保持原行为：`retrieval_query` 是可选的检索原文；省略时使用 `question` 原文做字面匹配。
- `semantic` 模式先对问题做嵌入，再在已建立的语义候选向量中排序；`retrieval_query` 不参与语义检索。语义索引缺失、过期或模型不可用时返回明确错误，**不会退回**字面检索或无依据的自由回答。
- 语义候选同样来自存储的页面文本，引用字段由服务端按 `page_text[start:end] == source_text` 校验后重建；`support_kind = "semantic_candidate"`，并附带 `retrieval_kind = "semantic"` 与 `score`，与字面命中的 `retrieval_match` / `sentence` 明确区分。
- 无命中时返回 HTTP 200、`status = "insufficient_evidence"`、`citations = []`，且不调用模型。
- 证据展示模式返回 `status = "evidence_found"` 与服务端固定说明。
- 生成模式把服务端证据（`E1`、`E2`…）交给模型；模型只能返回 `answer` 与 `citation_ids`。未知编号、非法 JSON、超长或空回答会被拒绝；引用字段全部按服务端记录重建。
- 模型返回空 `citation_ids` 时丢弃其文本，返回 `insufficient_evidence`。
- 引用不包含 PDF 几何；点击引用需调用 `POST /evidence/locate`。
- 每个引用包含 `support_text` / `support_start` / `support_end` / `support_kind`：优先定位命中所在的完整句子（`sentence`）；跨行、句子边界不清或 Chunk 边界截断时回退为关键词命中（`retrieval_match`）。两种区间都是存储页面文本的精确切片，并位于同一 Chunk 内；句子切分只是标点启发式，不代表事实已被证明。
- 错误语义：404 文档不存在；409 未索引；422 请求非法；503 生成模式未启用/未配置；504 模型超时；502 模型服务失败或响应无法校验。

响应示例（生成模式成功）：

```json
{
  "document_id": "6f1c9e2a-1b2c-4d5e-8f90-123456789abc",
  "question": "论文提出的方法有哪些特点？",
  "retrieval_query": "proposed method",
  "match_case": false,
  "answer_mode": "grounded_generation",
  "retrieval_mode": "literal",
  "status": "answered",
  "answer": "模型生成的回答……",
  "total_matches": 3,
  "citations": [
    {
      "citation_id": "E1",
      "document_id": "6f1c9e2a-1b2c-4d5e-8f90-123456789abc",
      "chunk_id": "chk_...",
      "page_index": 0,
      "page_number": 1,
      "chunk_index": 0,
      "match_start": 120,
      "match_end": 135,
      "hit_text": "proposed method",
      "chunk_text": "...",
      "extraction_version": "v1",
      "support_text": "The proposed method has three stages: ingestion, retrieval, and verification.",
      "support_start": 116,
      "support_end": 193,
      "support_kind": "sentence"
    }
  ]
}
```

引用校验只保证回答关联到真实检索证据，不保证回答内容正确。已使用本机 Ollama `qwen3:8b`（OpenAI 兼容端点）完成少量真实联调；Qwen3 默认 thinking 会占用输出 token，联调时设置 `RAG_ANCHOR_LLM_REASONING_EFFORT=none`。自动化测试仍只使用 Fake/Mock Provider。

语义检索的真实 embedding 调用与真实召回率评测**未执行**：本机 `ollama list` 仅有 `qwen3:8b`，没有 `bge-m3`，本轮未下载模型。离线测试与 `scripts/evaluate_semantic_retrieval.py --provider fake` 只验证工程管线（候选切分、偏移校验、归一化、排序、引用定位），数值不代表语义检索质量。安装授权模型后可运行：

```bash
RAG_ANCHOR_EMBEDDING_ENABLED=1 \
RAG_ANCHOR_EMBEDDING_BASE_URL=http://127.0.0.1:11434 \
RAG_ANCHOR_EMBEDDING_MODEL=bge-m3 \
uv run python scripts/evaluate_semantic_retrieval.py --provider auto
```

### 文件读取

```http
GET /api/v1/documents/{document_id}/file
```

成功响应直接返回已上传 PDF 的原始字节：

```http
HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: inline; filename="document.pdf"
```

- 路径只由服务端根据 UUID 与存储目录生成，客户端不能传入磁盘路径。
- 文件名固定为 `document.pdf`，不使用上传时的原始文件名。
- 使用 `FileResponse` 流式返回，不会删除、修改或重新登记文档。
- 前端可用该接口把 PDF 交给 PDF.js 加载，再做坐标高亮。

示例：

```bash
curl -OJ http://127.0.0.1:8000/api/v1/documents/<document_id>/file
```

### 错误状态码

| 场景                               | 状态码 | `detail`                                     |
| ---------------------------------- | ------ | -------------------------------------------- |
| 上传超限                           | 413    | `Uploaded file exceeds the maximum size ...` |
| 非法或损坏 PDF                     | 400    | `Uploaded file is not a valid PDF` 等        |
| 文档不存在                         | 404    | `Document not found`                         |
| `document_id` 不是 UUID            | 422    | FastAPI 校验错误                             |
| `query` 缺失、类型错误、空或纯空白 | 422    | FastAPI 校验错误                             |
| 已登记文档的 PDF 缺失或不可读      | 500    | `Stored PDF is missing or unreadable`        |
| 已登记文档的 PDF 无法解析          | 500    | `Stored PDF could not be searched`           |
| 索引时 PDF 缺失或不可读            | 500    | `Stored PDF is missing or unreadable`        |
| 索引时 PDF 无法抽取文本            | 500    | `Stored PDF could not be indexed`            |
| 索引写入 SQLite 失败               | 500    | `Document index could not be stored`         |
| 问答时文档未索引/索引版本不可用    | 409    | `Document has not been indexed` 等           |
| 语义索引缺失、过期或为空           | 409    | `Semantic index ...`                         |
| 语义检索/embedding 未启用或未配置  | 503    | `Embedding service ...`                      |
| embedding 请求超时                 | 504    | `Embedding request timed out`                |
| embedding 服务失败或返回非法向量   | 502    | `Embedding service ...`                      |
| 生成模式未启用或未配置             | 503    | `Grounded generation ...`                    |
| 模型请求超时                       | 504    | `LLM request timed out`                      |
| 模型服务失败或引用校验失败         | 502    | `LLM service request failed` 等              |

错误响应不会包含服务器绝对路径、存储目录或内部堆栈。

## 坐标契约

所有矩形坐标为 **PDF user space 绝对坐标**（单位 points，1 pt = 1/72 inch）：

- 原点在页面左下角，x 向右、y 向上。
- 不减去 CropBox 原点，也不把页面旋转烘焙进矩形。
- 页面尺寸、旋转与 CropBox 通过 `page` 字段独立给出。
- 前端由 PDF.js viewport 负责映射到屏幕像素。

详见 [`docs/adr/0001-coordinate-contract.md`](../docs/adr/0001-coordinate-contract.md)。

## 约束

- 禁止 `pip install` / `uv pip install`，依赖一律通过 `uv add` 管理。
- 不使用系统 Python 或 CALF 环境。
- PDF 引擎优先 `pypdfium2`；PyMuPDF 在许可证确认前不安装。
- 测试 PDF 在运行时动态生成，不提交二进制 PDF。
- 模型测试只注入 Fake/Mock Provider，不使用真实网络、模型权重或 GPU；`models/` 与常见权重/密钥文件不提交 Git。
- Mock 结果不得描述为真实生成效果；本机 Ollama 的少量联调结果也只代表该模型与当前测试样本。
