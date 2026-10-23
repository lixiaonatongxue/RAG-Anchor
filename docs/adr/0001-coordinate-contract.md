# ADR-0001：坐标契约

- 状态：Accepted；M0-02 已验证基础语义，M0-03.1 已统一生产适配器为 PDF user space 原始坐标
- 日期：2026-10-18
- 决策范围：PDF 几何数据的权威表示、派生表示与前端渲染职责

## 背景

本工具需要在 PDF 上精确高亮证据文字。坐标一旦定义不清，会在以下场景出错：

- 同一文字在页面中出现多次（多命中）
- 文字跨行、倾斜、旋转（需要 quads 而非单纯 bbox）
- 页面存在 `/Rotate`、`/CropBox` 与 `/MediaBox` 不一致
- 前端缩放、翻页、设备像素比变化

因此必须先固定“什么是事实、什么是派生、谁负责变换”。

## M0-01 原稿中被实测推翻的假设

M0-01 的初版 ADR 曾规定：

1. 左上角为原点、y 向下，并称与 PyMuPDF 页面坐标一致；
2. 坐标相对于 `/CropBox`。

M0-02 实测（`pypdfium2 5.14.0`）表明这两点在 PDFium API 上都不成立：

- reportlab 以基线 `(100, 200)` 绘制 12 pt 文本时，`get_charbox()` 返回 `(100.168, 200.0, 107.164, 208.616)`。`bottom == 200` 且 `top > bottom`，证明原点在左下、y 向上。
- CropBox 设为 `(50, 100, 400, 500)` 后，`get_charbox()` 仍返回绝对 user space 坐标（`quick` 左边界约 `124.432`），并未自动扣除 CropBox 原点。

因此本 ADR 以 PDF user space 原始矩形为权威表示。生产适配层直接输出 `get_charbox()` 的绝对坐标，**不**在服务内扣除 CropBox 原点；CropBox 仅作为页面元数据随结果返回，归一化等变换属于派生表示。这是对原契约的更正；证据固化为 `backend/tests/unit/test_geometry.py` 与 `backend/tests/unit/test_anchor_service.py`。

## 决策

### 1. 权威数据：PDF user space 原始矩形 + quads

后端输出的**事实来源**是 PDF user space 中以 points 为单位的原始几何数据：

- 单位：PDF points（1 pt = 1/72 inch），浮点数。
- 原点与轴向：**页面左下角为原点，x 向右，y 向上**。这是 PDF user space 的定义，已由 M0-02 实测确认（见下文“已验证行为”）。
- 页面边界由该页 `/CropBox` 描述，但 `PdfRect` 不相对 CropBox 存储。`pypdfium2` 的 `get_charbox()` 返回**未扣除 CropBox 原点的绝对 user space 坐标**，生产适配层必须保持该值原样，**不得**在 `search_anchors()` 内执行 `x - cropbox.left` / `y - cropbox.bottom`。CropBox 原点保存在 `PageDescriptor.cropbox`，供前端或派生归一化使用。

```text
PdfRect = get_charbox()  # 绝对 user space，不做 CropBox 平移
```

- 旋转：`get_charbox()` 返回**未旋转** user space 坐标；`/Rotate` 不烘焙进字符矩形。页面旋转角、CropBox、页面尺寸作为 `PageDescriptor` 随结果返回。
- 每个命中同时给出：
  - `quads`：该命中每个文本行的四点多边形（权威，支持倾斜/旋转文字）；
  - `bbox`：所有 quad 的最小外接矩形（派生，便于滚动定位与粗筛）。
- MVP **不承诺任意旋转文字的精确四边形**。`/Rotate` 页面旋转与“文字自身被旋转绘制”是两件不同的事，后者尚未验证。

### 2. PageDescriptor 的尺寸语义

```text
PageDescriptor
- page_index: int          # 0-based，机器标识与数组下标
- width_pt: float          # CropBox 宽度：cropbox.right - cropbox.left
- height_pt: float         # CropBox 高度：cropbox.top - cropbox.bottom
- rotation: int            # 顺时针角度：0 / 90 / 180 / 270
- cropbox: [left, bottom, right, top]  # PDF user space points 中的绝对边界
```

约束：

- `width_pt` / `height_pt` 表示**未旋转**的 CropBox 尺寸，用于解释同一页面上的原始坐标。
- **不得**直接用 `PdfPage.get_size()` 填充 `width_pt` / `height_pt`。实测 `get_size()` 已同时应用 CropBox 与 `/Rotate`：612×792 页面加 `/Rotate 90` 后返回 `(792, 612)`；CropBox 为 350×400 且 `/Rotate 90` 时返回 `(400, 350)`。
- 若前端或 API 需要“旋转后的显示尺寸”，应显式派生 `display_width_pt` / `display_height_pt`，不得与 `width_pt` / `height_pt` 混用。
- `PdfPage.get_cropbox()` 在未定义 CropBox 时回退到 MediaBox（`fallback_ok=True`）。

### 3. 旋转语义

- `PdfPage.get_rotation()` 返回**顺时针**页面旋转角度（度）。
- `/Rotate` 不改变文本可检索性，也不改变 `get_charbox()` 的坐标值；坐标始终是未旋转 user space。
- `/Rotate` 会改变 `get_size()` 与 PDF.js `viewport` 的显示尺寸，前端必须使用页面实际旋转角做显示变换。
- 旋转页面上的文本提取顺序可能改变。M0-02 实测 `/Rotate 90` 时，PDFium 返回的文本顺序与未旋转视觉阅读顺序不同；因此**不得把文本提取顺序直接当作阅读顺序**。

### 4. 归一化坐标：仅派生表示

为便于前端定位，可额外输出归一化矩形（派生表示，不是后端权威输出）：

```text
x_norm = (x_abs - cropbox.left) / page_width_pt
y_norm = (y_abs - cropbox.bottom) / page_height_pt
```

其中 `x_abs` / `y_abs` 是 `PdfRect` 的绝对 user space 坐标，`page_width_pt` / `page_height_pt` 是未旋转 CropBox 尺寸。归一化只在需要时派生，不得改变 `PdfRect` 的绝对坐标语义。

约束：

- 取值范围 `[0, 1]`，原点左下，y 向上。
- 归一化坐标**不是事实来源**，不落库为唯一数据，不从归一化坐标反推原始坐标。
- 仅在 API 边界按固定精度（6 位小数）输出。
- 若前端需要左上原点的 CSS 百分比，由前端按 `y_css = 1 - y_norm` 做显示层转换，不与后端事实混淆。

### 5. 渲染变换：PDF.js 负责

- PDF.js 的 `viewport` 负责 PDF user space 到画布像素的变换，包括 `/Rotate`、CropBox 与缩放。
- 后端**不输出像素坐标、不输出 DPI、不假设缩放比例**。
- 前端不允许自行用 pt→px 常数换算，也不允许修改已保存的几何数据。

### 6. 页码与顺序

- `page_index`：0-based，作为机器标识与数组下标。
- `page_number`：1-based，仅用于展示。
- 多命中排序：`page_index` 升序，同页按阅读顺序（自上而下、自左而右）稳定排序。阅读顺序算法尚未实现，见“未确定 / 未承诺”。

### 7. 零命中 / 多命中

- 零命中：返回空数组 `[]`，HTTP 200，不视为错误。
- 多命中：返回全部命中，每个命中独立携带页码与几何数据，不合并、不去重。
- 单命中：长度为 1 的数组，不特殊化返回结构。

## 已验证行为（M0-02）

验证环境：`pypdfium2 5.14.0`（PDFium 随包绑定），测试 PDF 由 `reportlab` 动态生成，旋转/CropBox 由 `pypdf` 写入。测试文件：

- `backend/tests/unit/conftest.py`
- `backend/tests/unit/test_search.py`
- `backend/tests/unit/test_geometry.py`

| 行为                 | 实测结果                                                                                                                                                                    | 固化测试                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 多页文档             | `len(PdfDocument)` 与页数一致；`document[0]` 为第 1 页                                                                                                                      | `test_pages_are_addressed_by_zero_based_index`                         |
| 文本提取             | `PdfTextPage.get_text_range()` 返回整页文本，行间为 `\r\n`                                                                                                                  | `test_character_count_matches_extracted_text`                          |
| 文本搜索 API         | `PdfTextPage.search(...)` 返回 `PdfTextSearcher`；`get_next()` 返回 `(start_index, count)`，末尾返回 `None`                                                                 | `test_search_returns_character_index_and_count`                        |
| 多命中               | 循环调用 `get_next()` 可取得全部命中，不合并、不去重                                                                                                                        | `test_search_finds_all_occurrences_and_is_case_insensitive_by_default` |
| 大小写               | `match_case=False` 为默认值，`quick` 命中 `QUICK`；`match_case=True` 仅命中大小写完全一致项                                                                                 | 同上                                                                   |
| 零命中               | `get_next()` 返回 `None`；适配层应转换为 `[]`                                                                                                                               | `test_search_returns_no_hits_for_absent_text`                          |
| 跨行搜索             | `search("gamma second")` 命中跨 `\r\n` 的文本；`get_text_range(start, count)` 返回 `"gamma\r\nsecond"`                                                                      | `test_search_matches_phrase_across_line_break`                         |
| 字符坐标原点         | reportlab 基线 `(100, 200)`、12 pt 的 `T` 返回 `(100.168, 200.0, 107.164, 208.616)`，即左下原点、y 向上                                                                     | `test_charbox_uses_pdf_user_space_with_lower_left_origin`              |
| 字符矩形合并         | `The` 三个字符合并为 `(100.168, 199.832, 120.196, 208.616)`                                                                                                                 | `test_merging_character_boxes_yields_word_bounding_box`                |
| CropBox 与尺寸       | CropBox `(50, 100, 400, 500)` 时 `get_size() == (350, 400)`                                                                                                                 | `test_cropbox_sets_page_size_but_charbox_stays_absolute`               |
| CropBox 与字符坐标   | `get_charbox()` 仍返回绝对 user space（`quick` 左边界约 `124.432`），不自动扣除 `(50, 100)`；显式扣除后左边界约 `74.432`（仅为 M0-02 引擎行为记录；生产适配器不执行该扣除） | 同上                                                                   |
| `/Rotate 90/180/270` | `get_rotation()` 返回精确角度；`get_size()` 返回旋转后尺寸；`get_charbox()` 与未旋转页面完全一致                                                                            | `test_rotation_is_reported_without_rotating_charboxes`                 |
| Rotate + CropBox     | CropBox 350×400、`/Rotate 90` 时 `get_size() == (400, 350)`                                                                                                                 | `test_rotation_and_cropbox_combine_in_reported_page_size`              |
| 字符计数             | 样例中 `count_chars() == len(get_text_range())`                                                                                                                             | `test_character_count_matches_extracted_text`                          |

这些测试只固化 M0-02 观察到的事实。生产 `search_anchors()` 已在 M0-03 基于这些事实实现，M0-03.1 将其坐标输出统一为绝对 user space。

## 未确定 / 未承诺

1. **任意旋转文字的精确四边形**：尚未用文字自身旋转（文本矩阵/`drawRotatedString` 等）的样例验证。MVP 不承诺该场景的 quad 精度。
2. **跨行命中的拆分规则**：M0-03 已按 `\r\n` 拆分并输出每行一个轴对齐 `PdfRect`；quads（倾斜文字）与更细的空格/软换行规则仍未确定。
3. **空白归一化**：`search("gamma second")` 可以跨 `\r\n` 命中，但多空格、制表符、软换行、连字符断词的行为未系统验证。`match_mode` 的 `exact` / `casefold` / 后续扩展尚未定稿。
4. **阅读顺序**：尚未实现同页命中排序；旋转页面的文本提取顺序可能变化，不能直接复用。
5. **CropBox 外字符**：尚未验证是否应过滤 CropBox 之外的文本命中。
6. **CropBox 继承**：`pypdfium2` 文档指出 `get_*box()`（除 `get_bbox()`）在 PDFium 5418 中不继承父节点；本阶段只测试了页级 CropBox。
7. **Unicode 覆盖**：`get_text_range()` 文档警告仅支持 UCS-2；CJK、RTL、连字、组合字符、U+10000 以上字符尚未验证。`get_text_bounded()` 提供完整 Unicode 支持，但未用于本阶段。
8. **搜索重叠**：默认 `consecutive=False` 会跳过重叠匹配；`consecutive=True` 仅确认 API 存在，未纳入契约。
9. **前端端到端**：PDF.js `viewport` 与后端 PDF user space 绝对坐标的联调尚未进行。
10. **加密 PDF、扫描件、XFA、OCR**：均未验证，且不属于当前阶段范围。

## 第三方许可记录

本阶段只记录事实，不构成法律结论，也不代表项目整体许可证问题已解决。

| 组件                          | 版本   | 元数据中的许可信息                                                           | 用途                            |
| ----------------------------- | ------ | ---------------------------------------------------------------------------- | ------------------------------- |
| `pypdfium2`                   | 5.14.0 | `BSD-3-Clause, Apache-2.0, dependency licenses`                              | 运行时 PDF 解析引擎             |
| PDFium（随 `pypdfium2` 绑定） | 随包   | PDFium 本身为 BSD-3-Clause，并捆绑第三方组件；分发时必须保留其许可与通知文件 | 运行时原生库                    |
| `reportlab`                   | 5.0.1  | BSD license                                                                  | 仅测试动态生成 PDF              |
| `pypdf`                       | 6.21.1 | BSD-3-Clause                                                                 | 仅测试写入 `/Rotate` 与 CropBox |

约束：

- 发布前必须从实际安装的 wheel 中提取并保留 PDFium 及其第三方组件的完整许可文本；本阶段尚未生成该清单。
- 项目最终许可证（MIT / Apache-2.0 等）尚未确定，当前不创建 `LICENSE` 文件。
- PyMuPDF 在许可证确认前不安装。

## 示例（示意，非最终 schema）

坐标值为 PDF user space 绝对坐标、左下原点、y 向上的 PDF points；前端可按需派生 CropBox 相对或归一化坐标；`bbox` / `quads` 为派生展示值。

```json
{
  "query": "evidence text",
  "match_mode": "exact",
  "hits": [
    {
      "page_index": 2,
      "page_number": 3,
      "matched_text": "evidence text",
      "quads": [
        [
          [72.0, 144.0],
          [240.0, 144.0],
          [240.0, 156.0],
          [72.0, 156.0]
        ]
      ],
      "bbox": [72.0, 144.0, 240.0, 156.0],
      "page": {
        "width_pt": 595.28,
        "height_pt": 841.89,
        "rotation": 0,
        "cropbox": [0.0, 0.0, 595.28, 841.89]
      }
    }
  ]
}
```

## M1-03B 复用说明（evidence/locate）

M1-03B 的精确证据定位（`POST /api/v1/documents/{document_id}/evidence/locate`）**完全复用**本契约，不引入任何新的坐标语义：

- `EvidenceLocationService` 复用 `anchor_service` 的 `match_rects()` 与 `build_page_descriptor()`（仅由内部私有名提升为公开名，行为逐字不变）。`rects` / `bbox` / `page` 的语义与 `search_anchors` 完全一致：绝对 user space、单位 points、不减 CropBox 原点、不烘焙 `/Rotate`、跨行每行一个矩形、`bbox` 由真实 `rects` 派生。
- 新增的映射只发生在**文本索引空间**，不触碰几何语义：Python `str` 偏移（存储页面文本，即 `get_text_range()` 返回值）→ PDFium 字符索引 → `get_charbox()` 几何。该映射显式且双向校验；边界无法精确表达时明确失败（HTTP 409），绝不返回近似矩形。
- 若命中区间**内部**包含 PDFium 无法表达为空段的字符，同样返回 409。实测这类字符的 `get_charbox()` 仍返回有效矩形（非 BMP 探针：`(107.94, 697.89, 113.94, 708.46)`），若包含在区间内会产生调用方未请求的高亮；命中之前或之后的空段由严格边界选择排除，仍可正常定位。
- 映射完成后额外校验 `get_text_range(char_start, char_count)` 与逐字符拼接结果一致；不一致说明 PDFium 在整段提取时重写或丢弃了字符，此时返回 409。
- `hit_text` 仅用于校验，不用于搜索；occurrence 由 `page_index + match_start + match_end` 唯一确定，不从重复文本推测。
- 本复用不改变本 ADR 的任何既有决策；`search_anchors` 的行为与错误语义保持不变。

## 后果

正向：

- 几何数据可跨渲染器、跨缩放、跨 DPI 复用。
- 多行/倾斜文字用 quads 表达，不会因 bbox 丢精度。
- 前端实现简单，避免重复实现 PDF 坐标变换。
- 旋转与 CropBox 语义已由可重复测试固化，不再依赖“碰巧对齐”。

代价：

- 响应体比单一 bbox 略大。
- 调用方若需要 CropBox 相对坐标，必须自行显式派生，且只能做一次；后端不隐式平移。
- `width_pt` / `height_pt` 与显示尺寸分离，前端需理解二者差异。
- 旋转文字、Unicode、阅读顺序等场景仍需后续验证，当前不能对外宣称完全支持。

## 待办

1. 生产适配器已实现：`get_charbox()` 的绝对 user space 坐标原样输出（M0-03 / M0-03.1），不做 CropBox 平移。
2. 定义跨行命中的 quad 分组规则并补测试。
3. 确定 `match_mode` 枚举与空白归一化规则。
4. 实现同页阅读顺序排序，并验证 `/Rotate` 页面。
5. 验证文字自身旋转、CJK/RTL、连字与组合字符。
6. 发布前生成并保留 PDFium 及 `pypdfium2` 的完整第三方许可清单。
