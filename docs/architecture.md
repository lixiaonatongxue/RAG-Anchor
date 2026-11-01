# 架构说明

## 1. 目标

给定 PDF 与一段证据文字，返回该文字在 PDF 中**全部**出现位置的页码与几何坐标，供前端精确定位与高亮。

```text
PDF + query
    │
    ▼
[文档接入] 上传 / 校验 / 存储 / 元数据
    │
    ▼
[文本检索] search_anchors(pdf_path, query, match_case)
    │            ├─ 零命中  → []
    │            ├─ 单命中  → [AnchorHit]
    │            └─ 多命中  → [AnchorHit, ...]
    ▼
[几何输出] page_index / rects / bbox / page
    │
    ▼
[前端渲染] PDF.js 渲染页面 → viewport 变换 PDF user space 坐标 → 绝对定位高亮
```

## 2. 组件边界

### 2.1 后端（`backend/`）

独立 uv 项目，FastAPI 应用。当前分层：

| 层          | 职责                                                                                      | 当前状态                                                                                                                                                                                                                            |
| ----------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `api/`      | HTTP 路由、请求/响应模型                                                                  | health、文档管理（M0-04A）、证据搜索（M0-04B）、文件读取（M0-05A）、索引入口（M1-03A）、Chunk 检索（M1-02）、精确证据定位（M1-03B）、证据展示/生成式问答                                                                              |
| `domain/`   | 与框架无关的领域模型（`AnchorHit`、`PdfRect`、`PageDescriptor`、`PageText`、`TextChunk`） | M0-03 已实现 `PdfRect` / `AnchorHit` / `PageDescriptor`；M1-01 已实现 `PageText` / `TextChunk`                                                                                                                                      |
| `services/` | PDF 解析、文本抽取与切分、索引、检索、证据定位、问答、文档管理                            | M0-03 `search_anchors`；M0-04A `DocumentService`；M1-01 文本抽取与 Chunk 切分；M1-02 `ChunkRetrievalService`；M1-03A `DocumentIndexService`；M1-03B `EvidenceLocationService`；`QuestionAnswerService`（两种模式，共用字面检索）         |
| `infra/`    | 本地文件存储、SQLite 元数据仓储、可注入模型 Provider                                      | M0-04A `DocumentRepository`；M1-01 新增页面文本、Chunk 与索引元数据表；`infra/llm.py` 提供 OpenAI 兼容 HTTP Provider 与 Disabled Provider                                                                                            |

运行时配置位于 `rag_anchor/config.py`（读取 `RAG_ANCHOR_DATA_DIR`、`RAG_ANCHOR_MAX_UPLOAD_BYTES` 及 `RAG_ANCHOR_LLM_*`）。模型相关配置默认关闭：即使填写了地址，未设置 `RAG_ANCHOR_LLM_ENABLED=true` 也不会发起任何模型请求。

依赖方向：`api → services → domain`，`infra` 只被 `services` 依赖。领域层不得 import FastAPI。

### 2.2 核心服务 API

```python
def search_anchors(
    pdf_path: str | Path,
    query: str,
    *,
    match_case: bool = True,
) -> list[AnchorHit]: ...
```

语义要求：

- 返回**所有**匹配位置，按 `(page_index, char_start)` 排序。
- 零命中返回空列表，不抛异常、不返回 `null`。
- 多命中必须全部返回，不得去重或只取首个。
- 每个命中携带其所在页的 `PageDescriptor`（尺寸、旋转、CropBox），使坐标可独立解释。
- `rects` 为逐行矩形，跨行命中保留多行矩形，不生成跨行大框。
- 所有矩形均为 PDF user space 绝对坐标（points）。

### 2.3 HTTP 接口（M0-04A / M0-04B / M0-05A / M1-02 / M1-03A）

| 方法   | 路径                                              | 说明                                          |
| ------ | ------------------------------------------------- | --------------------------------------------- |
| GET    | `/api/v1/health`                                  | 健康检查                                      |
| POST   | `/api/v1/documents`                               | 上传 PDF（multipart `file`）                  |
| GET    | `/api/v1/documents`                               | 分页列出文档                                  |
| GET    | `/api/v1/documents/{document_id}`                 | 查询单个文档元数据                            |
| DELETE | `/api/v1/documents/{document_id}`                 | 删除文档元数据与存储文件                      |
| POST   | `/api/v1/documents/{document_id}/anchors/search`  | 检索证据文字并返回全部命中位置                |
| POST   | `/api/v1/documents/{document_id}/index`           | 建立或重建页面文本与 Chunk 索引（M1-03A）     |
| POST   | `/api/v1/retrieval/search`                        | 在已索引 Chunk 中检索并返回来源偏移（M1-02）  |
| POST   | `/api/v1/documents/{document_id}/evidence/locate` | 将页面文本 span 定位为真实 PDF 几何（M1-03B） |
| POST   | `/api/v1/documents/{document_id}/qa`              | 单文档问答：证据展示（默认）或 AI 生成，引用由服务端校验 |
| GET    | `/api/v1/documents/{document_id}/file`            | 读取已上传 PDF 原始字节（内联展示）           |

索引入口 `POST /documents/{document_id}/index` 只调用 `DocumentIndexService.index_document()`：PDF 哈希、文本抽取、Chunk 切分、来源校验与单事务原子替换旧索引均不在路由层重写。重复调用会重建索引，相同 PDF 与配置下最终 Chunk 集合一致，不是缓存命中；重建失败时保留原索引。

搜索请求：

```json
{
  "query": "The proposed method",
  "match_case": true
}
```

搜索响应：

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

错误语义：

| 场景                                | 状态码 |
| ----------------------------------- | ------ |
| `document_id` 不是 UUID             | 422    |
| `query` 缺失、类型错误、空或纯空白  | 422    |
| 文档元数据不存在                    | 404    |
| 已登记文档的 PDF 缺失、损坏或不可读 | 500    |

零命中返回 HTTP 200，`total` 为 `0`，`results` 为 `[]`。错误响应不包含服务器绝对路径、存储目录或内部堆栈。

搜索路由是普通同步 `def`，由 FastAPI 线程池执行阻塞的 PDFium 解析，不阻塞事件循环。

### 2.4 文件读取（M0-05A）

```http
GET /api/v1/documents/{document_id}/file
```

返回已上传 PDF 的原始字节，供浏览器或 PDF.js 直接加载。实现复用 `DocumentService.get_document_path()`：路径始终由服务端 UUID 派生，客户端不能提交磁盘路径，也不能通过文件名选择文件。

响应头：

```http
Content-Type: application/pdf
Content-Disposition: inline; filename="document.pdf"
```

- `Content-Disposition` 固定为 `inline`，文件名固定为 `document.pdf`，不使用用户上传时的原始文件名。
- 使用 Starlette/FastAPI `FileResponse` 流式发送文件，不引入额外依赖。
- 读取是只读操作：不删除、不修改、不重新登记文档元数据。
- 错误语义：元数据不存在 → 404；非法 UUID → 422；元数据存在但文件缺失或不可读 → 500，且不泄露服务器路径。

### 2.5 精确证据定位（M1-03B）

M1-02 的 Chunk 检索只返回页面文本的 Python 偏移，不返回几何。M1-03B 新增独立端点：

```http
POST /api/v1/documents/{document_id}/evidence/locate
```

请求体：

```json
{
  "items": [
    {
      "page_index": 0,
      "match_start": 120,
      "match_end": 129,
      "hit_text": "evidence",
      "chunk_id": "chk_..."
    }
  ]
}
```

- `items` 长度 1..20，保持输入顺序，不静默去重；任一项失败则整个请求失败（all-or-nothing）。
- `match_start` / `match_end` 是**存储页面文本的 Python `str` 偏移**，与 `POST /api/v1/retrieval/search` 返回的偏移属于同一空间；它们不是 PDFium 字符索引，也不是 PDF 坐标。
- `hit_text` 仅用于校验，服务不会用它在 PDF 中重新搜索，也不会据此推测 occurrence。
- `chunk_id` 可选；提供时必须是同一文档、同一页，且 span 落在该 Chunk 范围内。

响应：

```json
{
  "document_id": "6f1c9e2a-1b2c-4d5e-8f90-123456789abc",
  "total": 1,
  "results": [
    {
      "page_index": 0,
      "page_number": 1,
      "match_start": 120,
      "match_end": 129,
      "hit_text": "evidence",
      "chunk_id": "chk_...",
      "char_start": 121,
      "char_count": 8,
      "matched_text": "evidence",
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

- `char_start` / `char_count` 属于 PDFium 字符索引空间；`rects` / `bbox` 是 PDF user space 绝对坐标（points），坐标契约与 M0-03 完全一致。
- `rects` 只来自真实 `get_charbox()`；无有效几何时明确失败，不生成近似矩形。

数据流：

```text
POST /retrieval/search → (page_index, match_start, match_end, hit_text, chunk_id)
        │
        ▼
POST /evidence/locate
        │  1. 只读校验：文档存在、index_metadata 存在、extraction_version 匹配、
        │     PDF SHA-256 匹配、涉及页文本与存储完全一致
        │  2. 每个涉及页构建一次 offset map：segments[i] = get_text_range(i, 1)
        │  3. Python span → PDFium [char_start, char_end)，并双向校验重建文本
        │  4. match_rects(text_page, char_start, char_count) → 真实逐行矩形
        ▼
page_index / page_number / char_start / char_count / rects / bbox / page
```

实现约束：

- 每个请求只打开一次 `PdfDocument`；每个涉及页面只构建一次 offset map，不按命中重复建图。
- 非 BMP 字符在 PDFium 中可能产生空段；映射依据累积偏移定位精确边界，边界落在无法表达的字符内部时返回 409，绝不返回近似范围。
- 若命中区间**内部**包含无法表达为空段的字符，同样返回 409：实测该字符的 `get_charbox()` 仍返回真实矩形，包含它会产生调用方未请求的高亮；位于命中之前或之后的空段由严格边界选择排除，不影响正常定位。
- 映射完成后还会校验 `get_text_range(char_start, char_count)` 与逐字符拼接结果一致；不一致时返回 409，避免 PDFium 在整段提取时重写或丢弃字符。
- 校验只读：不重新索引、不写数据库、不刷新 fingerprint。
- 错误语义：404 文档不存在；422 请求项与存储文本/Chunk 不一致；409 未索引、索引与当前 PDF 不一致、span 无法精确映射或无真实几何；500 PDF 缺失或不可读。

### 2.6 问答：证据展示与生成式回答

```http
POST /api/v1/documents/{document_id}/qa
```

请求体：

```json
{
  "question": "论文提出的方法有哪些特点？",
  "retrieval_query": "proposed method",
  "match_case": false,
  "max_citations": 5,
  "answer_mode": "evidence_only",
  "retrieval_mode": "literal"
}
```

`answer_mode` 可选，默认 `evidence_only`，另一个合法值是 `grounded_generation`。`retrieval_mode` 可选，默认 `literal`，另一个合法值是 `semantic`。

`literal` 模式复用 M1-02 的字面 Chunk 检索；`semantic` 模式使用 [2.7](#27-多语言语义检索m2-01) 的派生语义索引。两种检索模式与两种回答模式正交组合，但都共享同一套服务端引用校验与 `evidence/locate` 定位路径。

两种模式共享同一层检索与引用：

- `literal` 模式直接复用 M1-02 的字面 Chunk 检索。`retrieval_query` 是可选的检索原文；省略时使用 `question` 原文进行字面匹配。自然语言问题不会被语义理解。
- `semantic` 模式对 `question` 做 embedding，在派生语义索引中排序候选，再把候选映射回存储页面文本；`retrieval_query` 不参与语义检索。语义索引缺失、过期或模型不可用时返回明确错误，不回退到字面检索或无依据回答。
- 检索严格限定 `document_id`；引用只能来自该文档已持久化的 Chunk。服务端在返回前校验每个命中属于该文档，否则按内部错误处理，绝不跨文档泄露。
- 服务端按返回顺序给证据分配 `E1`、`E2`…；所有引用字段（`document_id`、`chunk_id`、`page_index`、`page_number`、`match_start`、`match_end`、`hit_text`、`chunk_text`、`extraction_version`）来自存储索引，不来自模型输出。
- 每个引用额外包含 `support_text`、`support_start`、`support_end`、`support_kind`。它们描述点击引用时应定位的精确页面文本区间：字面检索优先为命中所在的完整句子（`support_kind = "sentence"`）；跨行、句子边界不明确、命中跨越句子或 Chunk 边界截断时，回退为原关键词命中（`support_kind = "retrieval_match"`）；语义检索使用候选自身的精确页面区间（`support_kind = "semantic_candidate"`），并带 `retrieval_kind = "semantic"` 与 `score`。所有区间都满足 `page_text[support_start:support_end] == support_text`，且完全位于该引用的 Chunk 内。
- 句子切分是确定性标点启发式，仅用于选择可定位的原文区间，**不是**模型结论与句子之间事实支持关系的自动证明。
- 没有命中时返回 HTTP 200，`status = "insufficient_evidence"`、`citations = []`；两种模式都不会调用模型。
- 点击引用时前端调用 M1-03B `POST /evidence/locate`，使用 `page_index + support_start + support_end + support_text + chunk_id` 精确定位，并复用现有 PDF.js 高亮。原始 `match_*`/`hit_text` 保持不变，用于追溯检索命中。问答接口本身不返回 PDF 几何。

**`evidence_only`（默认）**

- 不调用任何语言模型。
- `status = "evidence_found"`；`answer` 是固定的服务端说明，明确标注“证据展示模式（未接入语言模型）”。
- 保持与上一版本完全兼容：请求未提供 `answer_mode` 时行为不变。

**`grounded_generation`（可选）**

1. 完成同样的字面检索并构造服务端引用。
2. 若没有证据，直接返回 `insufficient_evidence`，不调用模型。
3. 将问题与服务端证据发送给可注入的 `LLMProvider`（默认实现为 OpenAI 兼容 `/chat/completions`）。
4. 系统提示词与用户内容分离：`system` 只包含固定规则；`user` 是 JSON，包含 `question` 与 `evidence`（`id`、`page_number`、`text`）。PDF 文本被明确标记为不可信外部数据，其中的指令不得被执行。模型请求使用 OpenAI 兼容的 `response_format = {"type": "json_object"}`，把解码约束为单个 JSON 对象；该约束不替代服务端 schema 与证据记账校验。
5. 模型必须返回显式回答结构：

```json
{
  "answerable": true,
  "claims": [
    {"text": "一条实质性陈述。", "citation_ids": ["E1"]}
  ],
  "limitations": ["证据未覆盖的问题部分。"]
}
```

拒答必须使用：

```json
{"answerable": false, "claims": []}
```

6. 服务端从 `claims` 的 `text` 组装最终 `answer`；`limitations` 以服务端固定前缀“证据未覆盖:”附加。返回引用只包含被至少一条 claim 引用的证据，顺序按 claim 首次出现；模型提供的页码、偏移、Chunk ID、文档 ID 或几何字段一律忽略。未使用证据由服务端机械计算：`提供的候选 ID 集合 − claims 引用的 ID 集合`，只作为内部审计信息，不返回给前端。模型不再需要枚举未使用证据；若仍输出旧 `unused_evidence` 字段，服务端完全忽略，不将其作为事实支持、相关性或完整性的证明。
7. 服务端拒绝回答路径上的未知/非法引用编号、非 JSON 输出、缺失 `answerable`/`claims`、空 claim、空 `citation_ids`、超长回答等。`answerable = false` 时 `claims` 必须为空；拒答中的自由文本、遗留辅助字段（包括 `limitations`）均被丢弃且不校验结构，返回固定拒答。
8. 成功时 `status = "answered"`。结构校验只证明“每条展示陈述关联到本次检索到的真实证据”，**不证明事实蕴含、对象绑定、相关性或回答完整性**；`answerable = true` 是模型声明，不是正确性证明。服务端计算 unused evidence 只是集合差，同样不证明回答完整或事实正确。

引用与 claim 校验只保证结构关联和原文切片重建，不保证回答中的每个事实正确；提示词也无法消除幻觉，前端会明确提示需要人工核对。

**错误语义**

| 场景                                             | 状态码 |
| ------------------------------------------------ | ------ |
| 文档不存在                                       | 404    |
| 请求非法（含未知 `answer_mode`）                 | 422    |
| 文档尚未索引或索引提取版本不匹配                 | 409    |
| 生成模式未启用/未配置（`LLMDisabledError`）      | 503    |
| 模型请求超时                                     | 504    |
| 模型服务 HTTP 失败、响应无法解析或引用校验失败   | 502    |
| 检索返回跨文档引用（内部一致性错误）             | 500    |

问答不写入数据库，也不修改索引。错误响应不会包含 API Key、模型请求体或服务器绝对路径。

**模型接入状态**

- 配置通过 `RAG_ANCHOR_LLM_*` 环境变量完成；默认 `RAG_ANCHOR_LLM_ENABLED=false`。
- Provider 构造不探测网络；只有收到问答请求且两种前置条件都满足时才发送 HTTP。
- 自动化测试只使用 Fake/Mock Provider；另已在本机 Ollama `qwen3:8b` 的 OpenAI 兼容端点完成少量真实联调。
- Qwen3 默认 thinking 会先消耗输出 token，较小 `max_tokens` 下可能返回空 `content`。Ollama 端点接受 `reasoning_effort: "none"`；本服务通过 `RAG_ANCHOR_LLM_REASONING_EFFORT=none` 显式发送，未配置时不改变请求格式。
- 本次联调仅覆盖单页测试 PDF 与少量问答场景，不代表其他模型、复杂文档或生产规模的质量。

### 2.7 多语言语义检索（M2-01）

目标：让中文自然语言问题能够召回英文原文证据，同时保持所有引用、支持区间与高亮仍来自可校验的存储原文。

**组件**

- `EmbeddingSettings`：通过 `RAG_ANCHOR_EMBEDDING_*` 环境变量配置，默认关闭。记录模型名、运维修订标签、超时、批大小与最低余弦分数。`model_version` 只是人工换代标签，**不是模型身份证明**。
- `EmbeddingProvider`：可注入协议。生产实现为 Ollama 原生 `POST /api/embed`，并通过 `GET /api/tags` 解析模型稳定身份；禁用时返回明确的 `DisabledEmbeddingProvider`。测试使用确定性 Fake Provider，不访问网络。
- `SemanticCandidateBuilder`：从 `document_page_texts` 与 `document_chunks` 生成句子/受控窗口候选，保留精确 `page_index`、`chunk_id`、`start_offset`、`end_offset` 与 `source_text`。切分规则有独立版本号，规则变化即令旧索引失效。
- `SemanticIndexService`：建立和查询派生向量索引，校验来源、格式版本与模型身份，执行严格向量校验、L2 归一化、余弦 Top-K 与确定性排序。

**派生索引**

- 复用 `Document`、`PageText`、`TextChunk`、`DocumentIndexMetadata` 作为唯一原文来源；新增独立的 `semantic_index_metadata` 与 `semantic_candidates` 表，二者都通过外键 `ON DELETE CASCADE` 绑定文档。
- `semantic_index_metadata` 记录模型名、运维标签、模型 digest、向量维度、归一化规则（固定 `l2`）、索引格式版本、候选切分版本、源指纹、提取版本、候选数量与建立时间。
- `semantic_candidates` 记录候选 ID、Chunk、页码、Chunk 序号、起止偏移、原文与 L2 归一化后的向量。
- 重新建立字面索引时同一事务内删除语义索引；语义索引可随时删除或重建，但不是原文事实来源。
- 模型名/标签/digest、索引格式版本、候选切分版本、归一化规则、源指纹、提取版本、候选数量或向量维度与当前文档不一致时判定为过期，返回 409，不静默复用。
- 表结构通过 `CREATE TABLE IF NOT EXISTS` 加增量 `ALTER TABLE ADD COLUMN` 迁移；旧库缺少 digest/版本列时补默认值（空 digest、版本 0），旧索引按保守策略视为过期。迁移与重建失败都不删除原文或旧派生数据。

**模型身份**

- 建立索引前和查询前各调用一次 `GET /api/tags`，按模型名（含 `:latest` 归一）匹配条目并读取 digest；身份 = 模型名 + 已验证 digest，持久化真实 digest 而非仅可变 tag。
- 模型缺失、digest 缺失/为空或同名条目 digest 冲突时视为身份不可确认，返回 502；`/api/tags` 超时返回 504，HTTP/传输失败返回 502。不会把无法确认当作“未变化”。
- 查询前 digest 与索引记录不一致返回 409。身份请求不做长期缓存，避免绕过校验；每次显式操作只请求一次。

**候选与偏移**

1. 候选只从已存储的页面文本切分，优先完整句子，超长句子/无标点文本使用受控窗口。
2. 建索引时逐条校验 `page_text[start:end] == source_text` 且区间完整落在某个 Chunk 内；同时校验 `candidate.chunk_index == chunk.chunk_index`、`candidate.page_number == page.page_number` 等一致性；不满足即拒绝。
3. 嵌入只输出向量，用于相关性排序；**不生成、不猜测、不修正字符偏移**。
4. 读取向量时校验 BLOB 非空且长度为 4 的倍数、维度匹配、分量为有限 float32、范数非零且接近 1。非法向量使整次查询 409，不参与排序，也不返回部分结果。
5. 检索结果按 `(-score, page_index, chunk_index, start_offset, candidate_id)` 稳定排序，并列时不依赖数据库返回顺序。

**问答集成**

- `POST /documents/{id}/qa` 新增可选 `retrieval_mode: "literal" | "semantic"`，默认 `literal`，原请求行为不变。
- 语义路径：问题 → embedding → 候选排序 → 阈值过滤 → 服务端重建引用 → 可选的 `grounded_generation` → 现有 `evidence/locate` 高亮。
- 语义引用明确标注 `retrieval_kind = "semantic"`、`support_kind = "semantic_candidate"` 与 `score`，与字面命中区分；构建引用时再次校验候选与 Chunk/页面的页码、序号一致。
- 语义索引缺失/过期、embedding 未配置、超时或返回非法向量时返回可解释错误；**绝不回退**到字面检索或无依据的自由回答。文档不存在优先返回 404，不被 503 掩盖；建索引期间文档被删除返回 404，不产生 500。
- 前端在开始或重建语义索引时通过现有 request gate 使进行中的问答失效并清空旧答案/引用；重建完成或失败后均可重新提问，旧响应不能恢复旧引用。

**已知限制**

- 已用 `bge-m3:latest`（digest `7907646426070047…`）与 `qwen3:8b` 完成 v2.1 真实 grounded_generation 回归（一次固定合成数据集回归，不是生产质量保证）：42 条问题（29 可回答 + 13 不可回答），Semantic Recall@5 为 1.0、Top-1 为 0.8276（5 条 Top-1 排序错误），相关句 span recall@5 为 0.975、全部相关句命中率为 0.9655；28/29 可回答问题生成带引用答案、误拒答 1/29，13/13 不可回答问题固定拒答（0 条误答、0 条引用），本轮 HTTP 502 为 0/42；46 条引用全部通过 ID、原文重建和 PDF 定位检查（34 条落在标注相关句、12 条之外）。0.30 只作为当前评测配置，不是已验证的最佳阈值。
- 引用合法仍是来源校验，不是事实校验：自动辅助逐条审查认为 19/29 条答案严格干净，7/29 条超出问题范围，2/29 条不完整。该判断只针对 42 条合成数据，不是人工事实认证；真实浏览器已验证核心链路，但快速连续请求、模型故障与语义索引异常/过期场景尚未完整验收。
- M2-01.1 加强了提示词：要求保持对象/数量/时间/否定/条件绑定，区分操作结果与执行时间，多条件逐项核对，不增加未要求结论，只引用直接支持答案的证据。生成契约改为 `answerable` + `claims` + `limitations`；旧的 `answer`/`citation_ids` 两字段输出（包括空 `answer` + 空 `citation_ids`）不再被接受，缺失 `answerable`/`claims`、空 `claims` 或 `answerable = true` 却无有效引用等结构按 502 处理，标准拒答为 `answerable = false` 且 `claims = []`。服务端仍不校验回答与引用之间的事实蕴含关系。
- M2-01.1 起生成输出采用 `answerable` + `claims`（每条含 `text` 与 `citation_ids`）+ `limitations` 契约：服务端从 claims 组装回答；未使用证据由服务端用“提供候选 ID − claim 引用 ID”机械计算并记录为内部审计，模型不再枚举 `unused_evidence`。旧 `unused_evidence` 字段若仍出现会被完全忽略，不作为事实支持、相关性或完整性的证明。明确拒答（`answerable=false`、`claims=[]`）丢弃所有辅助字段且不校验其结构；拒答带 claims、引用未知 ID、空 claim、非法 JSON 等仍按 502 拒绝。结构校验仍不验证 claim 与证据的事实蕴含、对象绑定或完整性；旧 `answer`/`citation_ids` 输出按 502 拒绝。
- 离线比较 Top-5 与 Top-10：Top-10 只多召回 1 条相关句（`cross-page-upload-limits` 的 500 页句，rank 7），同时新增 149 条非相关候选；因此生产候选数仍为 5，阈值仍为 0.30。
- 已知限制：`keyword-login-method` 仍可能被误拒答；`cross-page-upload-limits` 的 500 页证据仍在当前 Top-5 之外；部分答案仍不完整或超出问题范围；浏览器 A–G 完整边界场景尚未全部验收（核心链路已由用户真实浏览器验证）；M2-02 首轮真实 PDF 探索评测已完成（5 题，标签未冻结）。结构合法与引用可定位均不等于事实正确或引用支持答案。
- 固定评测集要求至少 30 条问题、至少 10 条不可回答，并覆盖中文问题召回英文证据、英文改写、跨页证据、部分相关候选、同关键词干扰、相似主题不同结论、不可回答、Recall@5 与 Top-1 分歧。标签先于 embedding 固定；不可回答问题不进入 Recall@K 分母。
- 阈值扫描复用每条问题一次低阈值检索的分数，离线过滤得到 0.30–0.70 的指标，不额外调用 embedding；QA 和引用仍使用配置的默认阈值。
- `--provider fake` 使用确定性关键词向量，只证明工程管线可运行，不能代表真实语义检索质量。
- 暴力余弦排序适用于单文档、中小规模候选；不引入向量数据库或 ANN 索引。

### 2.8 前端

- 原生 TypeScript + Vite + PDF.js，不使用 React/Vue 或状态管理库。
- PDF.js 负责 PDF 用户空间 → 画布像素的变换。
- 前端消费后端返回的 PDF user space 绝对坐标；如需 CSS 百分比，由前端派生归一化坐标。
- 缩放、翻页、旋转不改变已保存的几何数据。
- 问答面板提供 `evidence_only` / `grounded_generation` 模式选择，以及 `literal` / `semantic` 检索方式选择；默认字面检索。每个异步请求都经过 request gate，切换文档、检索方式、回答模式或重新索引会使旧响应失效。
- 语义检索需要先点击「建立语义索引」；索引未建立、重建中或文档已切换时不允许提交，界面显示明确说明。语义索引失败不会自动改用字面检索。
- 生成模式失败时只显示错误提示，不显示伪造的回答；引用点击仍走现有 `evidence/locate` 与高亮路径。
- 所有服务端文本（包括模型输出与 PDF 内容）均以 `textContent` 渲染，不执行 HTML 或其中的指令。

## 3. 坐标与数据契约

见 [ADR-0001](adr/0001-coordinate-contract.md)。要点：

1. 权威数据是 PDF 原始几何坐标（points）与逐行矩形 `rects`。
2. 归一化坐标是派生表示，仅用于渲染定位。
3. 渲染变换由 PDF.js 完成，后端不输出像素坐标。
4. 页面旋转必须显式记录，不能在坐标中静默“烘焙”。

## 4. 存储

MVP 使用本地文件系统与标准库 SQLite：

```text
backend/.data/                 # 可由 RAG_ANCHOR_DATA_DIR 覆盖
├── documents/                 # {document_id}.pdf，磁盘文件名只由 UUID 生成
├── tmp/                       # 上传中的临时文件，验证通过后原子移入 documents/
└── metadata.sqlite3           # 文档元数据、页面文本、Chunk 与派生语义索引
```

`metadata.sqlite3` 同时保存原文派生数据（`document_page_texts`、`document_chunks`、`document_index_metadata`）与可删除/重建的语义索引（`semantic_index_metadata`、`semantic_candidates`）。语义索引不是事实来源，删除或重建不影响原文、Chunk 与字面检索。

`.data/` 整体不提交 Git。用户上传的原始文件名只作为展示元数据保存，绝不参与磁盘路径拼接。上传先写入 `tmp/`，用 PDFium 实际打开验证，再 `os.replace()` 原子移动到 `documents/`，最后写入 SQLite；数据库失败时删除已移动的文件。

## 5. 非目标（第一版）

- 仓库内置模型权重、推理框架或 GPU 推理（生成式问答与 embedding 均通过 HTTP 调用本机服务，默认关闭）
- 大规模向量数据库、ANN 索引、rerank、查询改写、多轮对话、跨文档检索
- 自动生成锚点（引用仍来自存储原文与可校验偏移）
- OCR、扫描件
- GPU / PyTorch
- 用户体系、多租户、对象存储、任务队列
- 标注编辑、协作、导出 PDF

## 6. 阶段划分

| 阶段    | 内容                                                                          |
| ------- | ----------------------------------------------------------------------------- |
| M0-01   | 仓库骨架、FastAPI health、测试与工具链（已完成）                              |
| M0-02   | `pypdfium2` 技术验证：文本搜索、字符坐标、`/Rotate` 与 CropBox 样例（已完成） |
| M0-03   | `search_anchors` 精确匹配、零/单/多命中测试（已完成）                         |
| M0-03.1 | 坐标契约一致性审查与修正（已完成）                                            |
| M0-04A  | PDF 上传与文档管理 API（已完成）                                              |
| M0-04B  | 证据搜索 API：`POST /documents/{id}/anchors/search`（已完成）                 |
| M0-05A  | PDF 文件读取 API：`GET /documents/{id}/file`（已完成）                        |
| M1-01   | 页面文本抽取与 Chunk 切分，持久化可追溯来源（已完成）                         |
| M1-02   | Chunk 检索 API：`POST /retrieval/search`（已完成）                            |
| M1-03A  | 索引入口：`POST /documents/{id}/index`（已完成）                              |
| M1-03B  | 精确证据定位：`POST /documents/{id}/evidence/locate`（已完成）                |
| M1-04   | 前端 Chunk 检索、证据问答与 PDF.js 高亮（已完成）                             |
| M1-05   | 证据展示问答 MVP（已完成并提交）                                              |
| M1-06   | 可选生成式问答应用侧接入、引用校验与前端模式选择（已完成；自动化 Mock + 本机 Ollama qwen3:8b 少量联调） |
| M2-01   | 多语言语义检索 MVP：可注入 embedding、派生向量索引、语义问答与前端检索方式选择（已完成并提交） |
| M2-02   | 首轮真实 PDF 探索评测与证据选择实验；B1 保留为实验、B2 未通过验收且未启用（探索题标签未冻结） |
