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
[文本检索] search_anchors(pdf_path, query, match_mode)
    │            ├─ 零命中  → []
    │            ├─ 单命中  → [AnchorHit]
    │            └─ 多命中  → [AnchorHit, ...]
    ▼
[几何输出] page_index / quads / bbox / page_descriptor
    │
    ▼
[前端渲染] PDF.js 渲染页面 → 归一化坐标 → 绝对定位高亮
```

## 2. 组件边界

### 2.1 后端（`backend/`）

独立 uv 项目，FastAPI 应用。规划中的分层：

| 层          | 职责                                                          | 当前状态              |
| ----------- | ------------------------------------------------------------- | --------------------- |
| `api/`      | HTTP 路由、请求/响应模型                                      | 仅有 `/api/v1/health` |
| `domain/`   | 与框架无关的领域模型（`AnchorHit`、`Quad`、`PageDescriptor`） | 尚未实现              |
| `services/` | PDF 解析、文本检索、坐标变换                                  | 尚未实现              |
| `infra/`    | 本地文件存储、配置                                            | 尚未实现              |

依赖方向：`api → services → domain`，`infra` 只被 `services` 依赖。领域层不得 import FastAPI。

### 2.2 核心 API（规划）

```python
def search_anchors(
    pdf_path: str,
    query: str,
    match_mode: str = "exact",
) -> list[AnchorHit]: ...
```

语义要求：

- 返回**所有**匹配位置，按 `(page_index, 阅读顺序)` 排序。
- 零命中返回空列表，不抛异常、不返回 `null`。
- 多命中必须全部返回，不得去重或只取首个。
- 每个命中携带其所在页的 `PageDescriptor`（尺寸、旋转、CropBox），使坐标可独立解释。

### 2.3 前端（后续阶段）

- React + TypeScript + Vite + PDF.js。
- PDF.js 负责 PDF 用户空间 → 画布像素的变换。
- 前端只消费归一化坐标，用 CSS 百分比定位高亮层。
- 缩放、翻页、旋转不改变已保存的几何数据。

## 3. 坐标与数据契约

见 [ADR-0001](adr/0001-coordinate-contract.md)。要点：

1. 权威数据是 PDF 原始几何坐标（points）与 quads。
2. 归一化坐标是派生表示，仅用于渲染定位。
3. 渲染变换由 PDF.js 完成，后端不输出像素坐标。
4. 页面旋转必须显式记录，不能在坐标中静默“烘焙”。

## 4. 存储

MVP 使用本地文件系统：

```text
data/
├── uploads/     # 原始 PDF，按文档 ID 存放
└── derived/     # 渲染缓存、索引（可重建）
```

`data/` 整体不提交 Git。文档元数据第一版可用 JSON 侧车文件，暂不引入数据库。

## 5. 非目标（第一版）

- LLM 问答、自动生成锚点
- embedding / 向量检索 / rerank
- OCR、扫描件
- GPU / PyTorch
- 用户体系、多租户、对象存储、任务队列
- 标注编辑、协作、导出 PDF

## 6. 阶段划分

| 阶段  | 内容                                                                          |
| ----- | ----------------------------------------------------------------------------- |
| M0-01 | 仓库骨架、FastAPI health、测试与工具链（已完成）                              |
| M0-02 | `pypdfium2` 技术验证：文本搜索、字符坐标、`/Rotate` 与 CropBox 样例（已完成） |
| M0-03 | `search_anchors` 精确匹配、零/单/多命中测试                                   |
| M0-04 | 上传与文档管理 API                                                            |
| M0-05 | 前端 PDF.js 渲染与高亮                                                        |
