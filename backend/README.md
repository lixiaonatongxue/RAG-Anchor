# RAG-AnchorLab Backend

独立 uv 项目（Python 3.11），FastAPI 后端。当前提供健康检查、PDF 文档管理、证据搜索与 PDF 文件读取接口。

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
| POST   | `/api/v1/retrieval/search`                       | 在已索引 Chunk 中检索并返回来源偏移         |
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
