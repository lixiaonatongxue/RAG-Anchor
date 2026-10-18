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

| 层          | 职责                                                             | 当前状态                                                       |
| ----------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| `api/`      | HTTP 路由、请求/响应模型                                         | health、文档管理（M0-04A）、证据搜索（M0-04B）                 |
| `domain/`   | 与框架无关的领域模型（`AnchorHit`、`PdfRect`、`PageDescriptor`） | M0-03 已实现 `PdfRect` / `AnchorHit` / `PageDescriptor`        |
| `services/` | PDF 解析、文本检索、文档管理                                     | M0-03 已实现 `search_anchors`；M0-04A 已实现 `DocumentService` |
| `infra/`    | 本地文件存储、SQLite 元数据仓储                                  | M0-04A 已实现 `DocumentRepository`                             |

运行时配置位于 `rag_anchor/config.py`（读取 `RAG_ANCHOR_DATA_DIR` 与 `RAG_ANCHOR_MAX_UPLOAD_BYTES`）。

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

### 2.3 HTTP 接口（M0-04A / M0-04B）

| 方法   | 路径                                             | 说明                           |
| ------ | ------------------------------------------------ | ------------------------------ |
| GET    | `/api/v1/health`                                 | 健康检查                       |
| POST   | `/api/v1/documents`                              | 上传 PDF（multipart `file`）   |
| GET    | `/api/v1/documents`                              | 分页列出文档                   |
| GET    | `/api/v1/documents/{document_id}`                | 查询单个文档元数据             |
| DELETE | `/api/v1/documents/{document_id}`                | 删除文档元数据与存储文件       |
| POST   | `/api/v1/documents/{document_id}/anchors/search` | 检索证据文字并返回全部命中位置 |

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

### 2.4 前端（后续阶段）

- React + TypeScript + Vite + PDF.js。
- PDF.js 负责 PDF 用户空间 → 画布像素的变换。
- 前端消费后端返回的 PDF user space 绝对坐标；如需 CSS 百分比，由前端派生归一化坐标。
- 缩放、翻页、旋转不改变已保存的几何数据。

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
└── metadata.sqlite3           # 文档元数据
```

`.data/` 整体不提交 Git。用户上传的原始文件名只作为展示元数据保存，绝不参与磁盘路径拼接。上传先写入 `tmp/`，用 PDFium 实际打开验证，再 `os.replace()` 原子移动到 `documents/`，最后写入 SQLite；数据库失败时删除已移动的文件。

## 5. 非目标（第一版）

- LLM 问答、自动生成锚点
- embedding / 向量检索 / rerank
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
| M0-05   | 前端 PDF.js 渲染与高亮（未开始）                                              |
