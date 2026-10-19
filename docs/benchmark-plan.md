# RAG-Anchor PDF 证据锚定质量评测体系设计

- 文档编号：M0-06A
- 状态：**设计稿，尚未实现**
- 日期：2026-10-18
- 适用对象：后端 `search_anchors()` 与 `POST /api/v1/documents/{document_id}/anchors/search`
- 不适用对象：前端像素级渲染、OCR、LLM 问答、生产容量规划

> 本文只定义评测目标、能力边界、数据集结构、指标、真值方案、目录结构、分阶段计划、风险与最小验收标准。
> 本文不声称任何实测性能数据；所有阈值均为**待校准的设计参数**，不是测量结果。

---

## 0. 结论摘要

1. 评测对象是**证据锚定质量**：是否找到全部命中、页码是否正确、矩形是否落在正确文字上、跨行是否保留逐行矩形、坐标是否符合 absolute PDF user space 契约。
2. 真值必须独立于被测 PDFium。采用四级真值：`T-DET`（由数据集规格推导的命中数与页码）、`T-LAYOUT`（由绘制原点与 reportlab 字体度量推导的布局框）、`T-MANUAL`（小规模人工标注）、`T-CROSS`（可选跨实现验证）。**禁止把 `get_charbox()` / `search_anchors()` 的输出复制为预期结果。**
3. 数据集规模适中：`v0-baseline` 22 个样本（M0-06B），`v0-extended` 追加 6 个样本（M0-06C），总计 28 个样本、目标不超过 60 页、生成 PDF 目标不超过 5 MiB。上述是设计目标，不是实测值。
4. 第一阶段只做服务层、纯 Python、无新依赖、无网络、无浏览器、无 LLM 的基线；第二阶段扩展双栏、表格、扫描件负向控制与独立验证；第三阶段做版本化与公开发布。
5. 第一阶段的质量判定以**检测指标 + 页码计数 + 中心误差 + 包含关系 + 逐行矩形结构**为主；IoU 只报告不单独设门，避免布局框与墨迹框差异造成系统性误判。

---

## 1. 评测目标

### 1.1 要回答的问题

评测体系需要回答以下问题，并按可解释的指标输出：

| 编号 | 问题 | 对应指标 |
|---|---|---|
| Q1 | 该找到的命中是否都找到了 | 命中召回率 |
| Q2 | 返回的命中是否都是真实命中 | 命中精确率 |
| Q3 | 每个命中的页码是否正确 | 每页计数一致率、页码混淆对 |
| Q4 | 矩形是否落在目标文字上 | 中心误差、布局框包含率、尺寸比 |
| Q5 | 矩形是否与文字的实际覆盖范围一致 | IoU（报告项）、覆盖率 |
| Q6 | 跨行证据是否保留逐行矩形 | 行召回率、行精确率、矩形数一致 |
| Q7 | 失败是否能被稳定归类 | 失败分类码 |
| Q8 | 运行耗时是否可复现地记录 | 首次调用、中位数、p95、最大值 |

### 1.2 不回答的问题

- 不评测 PDF 渲染是否正确；渲染由 PDF.js 负责。
- 不评测前端高亮是否视觉美观、是否自动居中。
- 不评测 OCR 质量；扫描件只作为“无文本层时不得误报”的负向控制。
- 不评测 LLM 问答、向量检索、rerank。
- 不做跨机器性能排名；耗时只做同机同次运行内的记录。
- 不做许可证法律结论；只在风险章节列出需要注意的字体与 PDFium 许可事项。

### 1.3 成功标准的分层

评测结果分为三层，避免把“基础设施错误”“能力不支持”“质量缺陷”混为一谈：

1. **Harness 正确性**：数据集能生成、规格能校验、指标能计算、报告能输出。
2. **能力符合性**：对“直接评测”场景，检测、页码、几何指标达到已校准阈值。
3. **能力边界**：对“不支持”场景，只验证安全失败（不崩溃、不误报、返回空结果或明确错误），不要求正确性。

---

## 2. 当前系统能力边界

### 2.1 已实现且可评测的能力

基于当前实现（`backend/src/rag_anchor/services/anchor_service.py`、`api/schemas.py`、`domain/models.py` 与 ADR-0001）：

| 能力 | 事实 |
|---|---|
| 输入 | 未加密、带文本层的 PDF；上传上限默认 20 MiB（服务层） |
| 检索 | 精确子串匹配；`match_case` 控制大小写；返回全部非重叠命中 |
| 零命中 | 返回空列表 / HTTP 200 + `total=0`，不是错误 |
| 多命中 | 同一页多次出现全部返回；不按 API 契约去重 |
| 跨页 | `page_index` 0-based，`page_number` 1-based，按 `(page_index, char_start)` 排序 |
| 跨行 | 命中跨行时返回多个逐行矩形；不合并为跨行大框 |
| 输出字段 | `matched_text`、`char_start`、`char_count`、`rects`、`bbox`、`page` |
| 页码 | `page_index` 与 `page_number` 同时返回 |
| 页面元数据 | `width_pt`、`height_pt`、`rotation`、`cropbox` |
| 坐标 | absolute PDF user space，单位 points，左下原点、y 向上 |
| CropBox | **不减去** CropBox 原点；`cropbox` 作为绝对边界返回 |
| 旋转 | `/Rotate` 通过 `rotation` 表达，**不烘焙**进矩形 |
| 页面尺寸 | `width_pt` / `height_pt` 是**未旋转** CropBox 尺寸 |
| 错误语义 | 非法 UUID/空 query → 422；文档不存在 → 404；存储 PDF 缺失/损坏 → 500 |

### 2.2 明确不支持或未承诺的能力

以下能力**不得**被评测为“应当正确”，只能作为边界或负向控制：

| 能力 | 当前状态 | 评测处理 |
|---|---|---|
| OCR / 扫描件文字识别 | 不支持 | 只验证无文本层时返回 0 命中且不崩溃 |
| 加密 PDF | 不支持 | 不纳入 v0；可作为 M0-06C 错误守护样本 |
| RTL 文字 | 未验证 | 不纳入正确性指标 |
| 复杂表格结构 | 无表格解析 | 只评表格内文本的检测与坐标，不评单元格语义 |
| 双栏阅读顺序 | 未实现阅读顺序 | 只评命中检测与列归属，不评返回顺序 |
| 任意角度旋转文字 | 未承诺 | 不纳入正确性指标 |
| Unicode 归一化 / 全角半角等价 | 未承诺 | 不作为失败依据 |
| 连字展开、软换行、连字符断词 | 未承诺 | 不作为失败依据 |
| 重叠匹配 | 默认非重叠，未在 API 契约中承诺 | 只做边界记录，不设门 |
| 非 BMP 字符查询 | 安全返回 0 命中 | 只验证不抛 `IndexError` |
| CropBox 外字符过滤 | 未实现 | 不纳入 |
| `quads` 字段 | **当前 API 未实现** | 不评测；不以 ADR 中的描述性 quads 作为成功标准 |
| 生产级并发 / 缓存 / 索引 | 未实现 | 不评测 |

> **重要契约说明**：ADR-0001 在描述中使用了 `quads` 一词，但当前实现的响应模型只有 `rects` 与派生 `bbox`，没有 `quads` 字段。本评测以**已实现的 `rects` / `bbox` / `page`** 为准，不推测未实现字段。

### 2.3 对评测的含义

1. `matched_text` 的换行、空格、大小写形式由 PDFium 文本提取决定，**不能作为唯一成功标准**。跨行命中可能返回 `"beta\r\ngamma"`，也可能因引擎规范化而不同；评测应以命中数量、页码、几何位置为准。
2. `char_start` / `char_count` 属于 PDFium 字符索引空间，历史上已出现与 Python 字符串索引不一致的问题（非 BMP 字符）。**不得**把 `char_start` 当作跨实现真值，也不得用它做唯一匹配键。
3. `bbox` 是 `rects` 的派生外包框，不是独立事实。评测只把它作为结构一致性检查。
4. 现有测试中的 `API 输出 == search_anchors() 输出` 以及 `_pdfium_union()` 对比属于**回归自洽性检查**，不是独立真值。本评测不得复用它们作为 expected。

---

## 3. 坐标契约与真值空间

### 3.1 固定契约

所有评测真值与系统输出必须在同一坐标空间比较：

```text
坐标系：absolute PDF user space
单位：points（1 pt = 1/72 inch）
原点：页面左下角
轴向：x 向右，y 向上
CropBox：不减去 cropbox.left / cropbox.bottom
旋转：/Rotate 不烘焙进矩形
页面尺寸：width_pt / height_pt = 未旋转 CropBox 的宽 / 高
```

### 3.2 真值框与预测框

- **真值布局框 `B_layout`**：由数据集规格中的绘制原点与字体度量推导。
- **预测矩形 `R`**：系统返回的 `rects` 中的单个矩形。
- **派生真值外包框**：多行真值框的并集，仅用于跨行聚合比较，不替代逐行比较。
- **`bbox` 一致性**：`bbox == union(rects)`，浮点误差 ≤ `1e-6`。这是结构不变量，不是独立精度真值。

### 3.3 单行布局框的计算

对一条绘制指令：

```text
segment = (x, baseline_y, text, font, size)
advance = pdfmetrics.stringWidth(text, font, size)      # points
ascent, descent = pdfmetrics.getAscentDescent(font, size)  # descent 为负
B_layout = [
    left   = x,
    right  = x + advance,
    bottom = baseline_y + descent,
    top    = baseline_y + ascent,
]
```

如果查询是 segment 的子串，给定 `char_offset` 与 `char_length`：

```text
prefix = text[:char_offset]
left   = x + stringWidth(prefix, font, size)
right  = left + stringWidth(query_text, font, size)
bottom = baseline_y + descent
top    = baseline_y + ascent
```

这是**布局框**，不是逐字墨迹框。它包含 side bearing、行距与字体上下沿，因此：

- 预测墨迹矩形通常应当**完全落在** `B_layout` 内；
- `IoU(R, B_layout)` 通常小于 1，**不能**单独作为通过/失败门；
- 单字符、CJK、带重音字符的差异更明显。

### 3.4 字体度量的可用性分级

| 字体类型 | 度量来源 | 真值等级 | 处理 |
|---|---|---|---|
| Helvetica / Times / Courier 等 base-14 | reportlab 内置 AFM 度量 | `layout-exact` | 可用严格阈值 |
| 嵌入式 TTF（如 DejaVuSans） | reportlab `TTFont` 度量 | `layout-exact` 或 `layout-band` | 需在 M0-06B 验证度量可用性 |
| CJK CID 字体（如 STSong-Light） | reportlab `UnicodeCIDFont` 度量可能不完整 | `layout-band` | 允许规格显式提供 advance/ascent/descent |
| 度量不可用 | 规格显式 `metrics` 覆盖 | `layout-band` | 必须标注 `source: manual` 与理由 |

如果字体度量不可用且没有显式覆盖，该样本标记为 `skipped`，不计入质量分母。

---

## 4. 真值独立性与独立验证方案

### 4.1 核心原则

> **被测系统的输出绝不能成为自己的预期结果。**

具体禁止：

| 反模式 | 为什么禁止 |
|---|---|
| 用 `get_charbox()` 生成 expected rect | 直接自我验证 |
| 用 `search_anchors()` 生成 expected 命中 | 直接自我验证 |
| 用 `_pdfium_union()` 作为 expected | 仍是 PDFium 输出 |
| 用 `char_start` / `char_count` 作为匹配键或页码真值 | 引擎索引空间，跨实现不稳定 |
| 用 `matched_text` 的精确形式作为唯一成功标准 | 文本规范化由引擎决定 |
| 用 API 与 service 输出一致作为精度证据 | 只是自洽性 |
| 只报 IoU 并据此判定质量 | 布局框与墨迹框天然不同 |

### 4.2 真值分级

| 等级 | 名称 | 来源 | 独立性 | 用途 |
|---|---|---|---|---|
| T-DET | 检测真值 | 数据集规格推导：文本、查询、出现次数、页码 | **完全独立于 PDF 引擎** | 命中数、精确率、召回率、每页计数 |
| T-LAYOUT | 布局真值 | 绘制原点 + reportlab 字体度量 | **独立于 PDFium**（同一字体度量体系） | 中心误差、包含率、尺寸比、逐行框 |
| T-MANUAL | 人工真值 | 人工在 PDF 查看器中确认命中与覆盖 | **独立于实现** | 小规模 spot-check、疑难样本 |
| T-CROSS | 跨实现真值 | 另一个解析器/渲染器（可选） | **独立于 pypdfium2** | M0-06C 可选验证 |
| T-INK | 墨迹真值 | PDFium `get_charbox()` | **不独立** | 仅允许作为回归快照，不进入质量指标 |
| T-SELF | 自洽真值 | API 与 service 对比 | **不独立** | 只用于回归，不作为质量证据 |

`T-INK` 与 `T-SELF` 可以在报告中出现，但必须标注 `consistency-only`，且不得计入 precision/recall/center error 的质量结论。

### 4.3 独立预检（在评测系统之前执行）

每个样本在进入质量指标前，必须先通过独立预检：

1. **结构预检**（`pypdf`）：
   - 页数等于规格；
   - `/Rotate` 等于规格；
   - `/CropBox` 等于规格（若规格声明）；
   - 生成文件可被 `pypdf` 打开。
2. **文本预检**（`pypdf`，独立于 PDFium）：
   - 期望页面的提取文本包含查询词（允许按规格声明的换行规范化）；
   - 若 `pypdf` 无法提取该字体文本，样本标记为 `preflight_partial`，**不得**用于检测门。
3. **规格自洽预检**：
   - 用纯字符串扫描规格自身的文本，重新计算每个查询的出现次数；
   - 必须等于规格声明的 expected 数量；
   - 防止查询词意外出现在非目标文本中造成假阳性。
4. **几何预检**：
   - 所有 `B_layout` 面积为正；
   - 同一页同查询的命中中心间距 ≥ 规格的 `min_separation_pt`；
   - 否则样本标记为 `ambiguous`，只计入页级检测，不计入坐标指标。

预检失败 → `F10_INVALID_FIXTURE`，不计入系统质量分母。

### 4.4 导入边界强制

为了让“真值独立”不依赖人工纪律，M0-06B 必须加入静态测试：

```text
允许 import pypdfium2 / rag_anchor 的模块：
  benchmarks/runner.py          # 唯一被测系统适配器
  benchmarks/cli.py             # 只调度，不计算真值

禁止 import pypdfium2 / rag_anchor 的模块：
  benchmarks/generate.py
  benchmarks/truth.py
  benchmarks/match.py
  benchmarks/metrics.py
  benchmarks/report.py
```

测试实现方式：用 `ast` 解析模块源码，检查 `Import` / `ImportFrom` 节点；发现禁止导入即失败。

### 4.5 人工验证（M0-06C）

- 每个场景至少 1 个样本做人工 spot-check。
- 人工只确认两件事：
  1. 预测矩形是否覆盖目标文字；
  2. 页码是否与视觉页面一致。
- 记录：样本 ID、查询、预测矩形、人工结论、日期。
- 人工验证不要求逐像素标注；它是独立于 PDFium 的定性对照，不替代 T-LAYOUT。

### 4.6 跨实现验证（M0-06C，可选）

- 若容器内存在独立工具（例如 Poppler 的 `pdftotext` / `pdftoppm`），可用于：
  - 文本层存在性交叉确认；
  - 光栅化后墨迹区域交叉确认。
- 必须满足：
  - 与 `pypdfium2` 不是同一代码库；
  - 不引入新的 Python 依赖；
  - 工具不可用时标记 `not_run`，**不得**写成 `passed`。
- 本设计不假设这些工具一定存在；M0-06C 开始时先做可用性探测。

---

## 5. 评测数据集设计

### 5.1 规模与生成原则

| 项目 | 设计目标 |
|---|---|
| `v0-baseline`（M0-06B） | 22 个样本，覆盖 A、B、C、E、F、H |
| `v0-extended`（M0-06C） | 追加 D、G、I 共 6 个样本，总计 28 个 |
| 每样本页数 | 1–3 页 |
| 总页数 | 目标 ≤ 60 页 |
| 生成 PDF 总大小 | 目标 ≤ 5 MiB |
| 生成方式 | `reportlab` 动态生成，`pypdf` 写入 `/Rotate` 与 `/CropBox` |
| 提交内容 | 只提交 JSON 规格与生成器源码；**不提交 PDF** |
| 随机性 | 不使用随机；如未来引入，必须固定 seed 并写入报告 |
| 字体 | base-14 优先；CJK/非 BMP 字体按可用性跳过 |
| 版权/隐私 | 不使用任何用户 PDF、外部下载 PDF 或受版权保护的内容 |

> 上述规模是设计上限，不是实测结果。

### 5.2 规格 Schema

规格使用 JSON，字段如下（示意）：

```json
{
  "schema_version": 1,
  "id": "A01",
  "scenario": "A",
  "title": "single-column single hit",
  "evaluation": "direct",
  "truth_level": "layout-exact",
  "page_size": [612.0, 792.0],
  "rotation": 0,
  "cropbox": null,
  "pages": [
    {
      "lines": [
        {
          "baseline_y": 200.0,
          "segments": [
            {"x": 100.0, "text": "The ", "font": "Helvetica", "size": 12.0},
            {"x": 124.0, "text": "zephyr", "font": "Helvetica", "size": 12.0},
            {"x": 165.0, "text": " is here.", "font": "Helvetica", "size": 12.0}
          ]
        }
      ]
    }
  ],
  "queries": [
    {
      "text": "zephyr",
      "match_case": true,
      "expected": [
        {
          "page_index": 0,
          "parts": [
            {
              "line_index": 0,
              "segment_index": 1,
              "char_offset": 0,
              "char_length": 6
            }
          ]
        }
      ]
    }
  ],
  "match_radius_pt": 12.0,
  "min_separation_pt": 24.0
}
```

字段语义：

| 字段 | 含义 |
|---|---|
| `evaluation` | `direct` / `boundary` / `unsupported` |
| `truth_level` | `layout-exact` / `layout-band` / `detection-only` / `boundary` |
| `pages[].lines[]` | 一条基线；同一基线可包含多个 segment |
| `segments[]` | 一次 `drawString`；`x` 是该段起点 |
| `expected[].parts[]` | 跨行命中的逐行部分；每部分对应一个 `B_layout` |
| `char_offset` / `char_length` | 在该 segment 内的字符偏移；用于子串查询 |
| `match_radius_pt` | 预测与真值的匹配半径 |
| `min_separation_pt` | 同页同查询命中之间的最小中心间距 |

设计要点：

- **segment 化绘制**让每条查询段的 `x` 已知，避免依赖前缀宽度。
- 跨行查询用多个 `parts` 表达，每行独立计算 `B_layout`。
- 不依赖 `char_start`；`char_offset` 只在规格内部用于计算布局框。
- 规格是文本文件，可审查、可版本化、可公开发布。

### 5.3 A–I 场景与样本定义

> 表中页码为 1-based；内部断言使用 `page_index = page_number - 1`。
> “真值来源”缩写：`DET` = T-DET，`LAY` = T-LAYOUT，`MAN` = T-MANUAL，`CROSS` = T-CROSS。
> 所有阈值均为待校准设计参数。

#### A. 单栏英文

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| A01 | 1 页，一行 `The zephyr is here.`，`zephyr` 为独立 segment | `zephyr` | 1 | 1 | DET + LAY | 命中数=1；页码=1；`center_error ≤ 1.0 pt`；`inside_ratio ≥ 0.999` | 漏检；矩形偏移；误匹配子串 |
| A02 | 1 页，3 行，每行 1 个 `quokka`，行距 60 pt | `quokka` | 3 | 1 | DET + LAY | 命中数=3；3 个矩形互不相同；按 `char_start` 升序 | 去重；只返回首个；行间串位 |
| A03 | 3 页，每页 1 个 `narwhal` | `narwhal` | 3 | 1,2,3 | DET + LAY | 每页 1 个；`page_index=[0,1,2]`；每页计数一致 | 跨页漏检；页码错位 |
| A04 | 1 页，`Quick` 与 `QUICK` 各 1 次 | `quick`，`match_case=true/false` | 1 / 2 | 1 | DET | 敏感=1；不敏感=2；不重复计数 | 大小写泄漏；重复计数 |
| A05 | 1 页，无 `zzzznotfound` | `zzzznotfound` | 0 | — | DET | `total=0`；`results=[]`；无异常 | 误报；抛异常 |

#### B. 中文与 Unicode

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| B01 | 1 页，`中文` segment + `quick` segment（x 显式给定） | `quick` | 1 | 1 | DET + LAY(band) | 命中数=1；矩形起点≈query 段起点；落在显式 cell 内 | CID 字体缺失；文本顺序变化；索引偏移 |
| B02 | 1 页，`café` segment + `quick` segment | `quick` | 1 | 1 | DET + LAY | 命中数=1；`center_error ≤ 1.0 pt`；无 500 | 重音字符导致索引偏移（历史 IndexError 场景） |
| B03 | 1 页，`𝄞` segment + `quick` segment，DejaVuSans；字体缺失则 skipped | `quick` | 1 | 1 | DET + LAY(band) | 命中数=1；无 500；矩形落在 cell 内 | 字体缺失；非 BMP 索引不一致 |
| B04 | 同 B03 | `𝄞` 或 `𝄞quick` | 0 | — | DET（边界） | 返回 `[]`；不抛 `IndexError`；HTTP 200 | 崩溃；误报 |

> B04 属于**边界样本**，只验证安全失败，不计入质量门。非 BMP 查询不在当前支持范围内。

#### C. 跨行文本

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| C01 | 1 页两行：line1 `alpha ` + `beta`；line2 `gamma` + ` delta` | `beta gamma` | 1 | 1 | DET + LAY | 命中数=1；`rects=2`；每行矩形落在对应行 band；`line_recall=1`；不得合并为 1 个矩形 | 合并为一个矩形；漏掉第二行；换行识别错误 |
| C02 | 3 页，每页同 C01 | `beta gamma` | 3 | 1,2,3 | DET + LAY | 每页 1 个命中、2 个矩形；每页计数一致 | 跨页结构错误 |
| C03 | 1 页三行，查询跨 3 行 | `beta gamma delta` | 1 | 1 | DET + LAY | `rects=3`；`line_recall=1`；逐行顺序一致（报告项） | 只返回 2 行；合并；漏行 |

> 不要求 `matched_text` 的换行符形式完全一致；跨行语义以“每行一个矩形”与几何位置为准。

#### D. 双栏排版

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| D01 | 2 栏，左栏 x=72，右栏 x=330；`orion` 左右各 1 次，y 不同 | `orion` | 2 | 1 | DET + LAY | 命中数=2；按矩形 x 中心归属左右栏正确；**不 gate 返回顺序** | 右栏漏检；两矩形落到同一栏 |
| D02 | 2 栏，左右各 2 次，y 交错 | `orion` | 4 | 1 | DET + LAY | 4 个矩形；列归属正确；不 gate 顺序 | 阅读顺序假设导致漏检；去重 |

> 当前不支持双栏阅读顺序。D 场景只评“是否找到、位置是否正确”，不评“先左后右”。

#### E. 非默认 CropBox

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| E01 | CropBox=(50,100,400,500)，`zephyr` 段起点 x=`x_seg`，基线 y=`y_seg` | `zephyr` | 1 | 1 | DET + LAY（绝对） | `rect.left ≈ x_seg`（不得为 `x_seg - 50`）；`width_pt=350`；`height_pt=400`；`cropbox` 精确 | 减了 CropBox 原点；页面尺寸错 |
| E02 | CropBox 同 E01，跨行查询 | `beta gamma` | 1 | 1 | DET + LAY | 2 个矩形均为绝对坐标；逐行 `center_error ≤ 1.0 pt` | 行矩形被平移 |

> E 场景是坐标契约的关键回归。任何 `left` / `bottom` 等于“绝对坐标减 CropBox 原点”的实现都应被判为失败。

#### F. 旋转页面

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| F01 | `/Rotate 90`，文本与 A01 相同 | `zephyr` | 1 | 1 | DET + LAY | 矩形与未旋转基准逐边差 ≤ 0.05 pt；`rotation=90`；`width_pt`/`height_pt` 未交换 | 旋转烘焙；尺寸交换 |
| F02 | `/Rotate 180` | `zephyr` | 1 | 1 | DET + LAY | 同上；`rotation=180` | 同上 |
| F03 | `/Rotate 270` | `zephyr` | 1 | 1 | DET + LAY | 同上；`rotation=270` | 同上 |
| F04 | `/Rotate 90` + CropBox=(50,100,400,500) | `zephyr` | 1 | 1 | DET + LAY | 矩形仍为未旋转绝对 user space；`rotation=90`；页面尺寸=350×400 | 旋转与 CropBox 组合错误 |

> 不 gate 文本提取顺序与 `char_start`；旋转页面的文本顺序可能变化。

#### G. 表格文字

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| G01 | 3×3 表格；`lumen` 在 cell(0,0) 与 cell(2,1) | `lumen` | 2 | 1 | DET + LAY | 2 个矩形分别落在对应 cell 的 x/y 区间；`center_error ≤ 1.0 pt` | 单元格文字漏检；坐标越界 |
| G02 | 表头与单元格各 1 次 `lumen` | `lumen` | 2 | 1 | DET + LAY | 同上；不要求识别表头/单元格语义 | 表头与单元格混淆（不要求结构识别） |

> G 场景只评文本层检索与坐标；不评表格结构、合并单元格、行列阅读顺序。

#### H. 重复词与大小写

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| H01 | 1 行 `quokka and quokka and quokka`，三个独立 segment，间隔 ≥ 2× 宽度 | `quokka` | 3 | 1 | DET + LAY | 3 个矩形；x 递增；互不重叠；`center_error ≤ 1.0 pt` | 去重；只返回首个；矩形重叠 |
| H02 | 5 行，每行 1 次 `quokka` | `quokka` | 5 | 1 | DET + LAY | 5 个矩形；行归属正确 | 漏行；行间串位 |
| H03 | `Quokka` / `QUOKKA` / `quokka` 各 1 次 | `quokka`，`match_case=true/false` | 1 / 3 | 1 | DET | 敏感=1；不敏感=3；不重复计数 | 大小写泄漏；重复计数 |
| H04 | `aaaa` 中查询 `aa` | `aa` | 边界：按非重叠语义记录 | 1 | DET（仅记录） | 不崩溃；返回非重叠矩形；**不设通过/失败门** | 对重叠匹配的期望错误；重复计数 |

> H04 是边界样本。当前 API 未承诺重叠匹配语义，因此只记录实际行为，不作为质量缺陷。

#### I. 扫描型 PDF

| ID | 内容 | 查询 | 期望命中 | 页码 | 真值来源 | 成功标准 | 可能失败原因 |
|---|---|---|---|---|---|---|---|
| I01 | 1 页，空白页，无文本对象 | `zephyr` | 0 | — | DET（负向） | `total=0`；`results=[]`；无 5xx | OCR 误触发；崩溃；误报 |
| I02 | 1 页，仅嵌入一张无文本图层的栅格图 | `zephyr` | 0 | — | DET（负向） | 同上；`pypdf.extract_text()` 为空 | 误把图像当文本；崩溃；误报 |

> I 场景不评测 OCR 质量。生成器只需保证页面没有文本对象；“图像内容是否真实可读”不是断言目标。栅格图可由标准库 `zlib` + `struct` 生成最小 PNG，或使用现有 `reportlab` 图像接口；不得引入新依赖。

### 5.4 可评测性矩阵

| 场景 | 命中检测 | 页码 | 几何坐标 | 结构/阅读顺序 | 结论 |
|---|---|---|---|---|---|
| A 单栏英文 | 直接评测 | 直接评测 | 直接评测 | 直接评测 | 基线核心 |
| B BMP CJK / 重音 | 直接评测 | 直接评测 | 布局带评测 | 直接评测 | 需 `layout-band` |
| B 非 BMP 前缀 | 直接评测（`quick`） | 直接评测 | 布局带评测 | 直接评测 | 需字体可用 |
| B 非 BMP 查询 | 仅负向安全 | — | — | — | **不支持** |
| C 跨行 | 直接评测 | 直接评测 | 直接评测（逐行） | 直接评测（逐行结构） | 基线核心 |
| D 双栏 | 直接评测 | 直接评测 | 直接评测 | **不支持**（阅读顺序） | 只评检测与列归属 |
| E CropBox | 直接评测 | 直接评测 | 直接评测（绝对空间） | 直接评测 | 坐标契约核心 |
| F 旋转 | 直接评测 | 直接评测 | 直接评测（未旋转空间） | 顺序不 gate | 只评几何与元数据 |
| G 表格 | 直接评测（文本层） | 直接评测 | 直接评测 | **不支持**（表格结构） | 只评文本与坐标 |
| H 重复/大小写 | 直接评测 | 直接评测 | 直接评测 | 直接评测 | 基线核心 |
| I 扫描件 | 仅负向 | — | — | **不支持**（OCR） | 负向控制 |

---

## 6. 评测指标

### 6.1 匹配算法

预测命中与真值命中的匹配必须与 `char_start` 解耦：

```text
对每一页 p：
  E_p = 该页 expected occurrences
  P_p = 该页 predicted hits

  候选对 (e, r)：
    - e 与 r 属于同一查询；
    - r 是单行矩形，或 e 的某个 part；
    - 中心距离 <= match_radius_pt。

  候选对仅在同一页内构造。
  在候选对上求解「一对一最小费用最大匹配」：先最大化匹配对数，
  再最小化匹配对的总中心距离（整数微点费用 round(中心距离 * 1_000_000)）。
  若某 expected 存在局部近邻歧义（最佳与次佳距离差 < 1.0 pt），
  或它属于等代价最优解之一（存在匹配数相同、总费用相同、但配对不同的最优解），
  则标记 AMBIGUOUS_MATCH：该对不计入坐标指标，但仍计入页级计数。
```

跨行命中的匹配：

1. 先按 occurrence 级 bbox 中心做同页最小费用最大匹配（规则见上）；
2. 再对预测矩形与各 `part` 的 `B_layout` 做一对一最小费用最大匹配
   （同样先最大化匹配数，再最小化总中心距离）；
3. 每个 `part` 最多一个预测矩形；输出顺序按 `part` 中心 y 降序保持稳定；
4. 未配对的预测矩形计入行级假阳性，未配对的 `part` 计入行级假阴性。

不使用的匹配键：

- `char_start` / `char_count`：引擎索引空间；
- `matched_text` 的精确换行/空格形式：引擎规范化；
- `bbox`：派生值。

### 6.2 检测指标

记：

```text
TP = 成功匹配的预测数
FP = 未匹配的预测数
FN = 未匹配的真值数

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

约定：

- 分母为 0 时输出 `null`（N/A），不输出 0 或 1；
- 同时报告 **micro-average**（汇总所有样本的 TP/FP/FN）与 **macro-average**（各样本指标的算术平均）；
- 报告 `n_defined`，说明参与平均的样本数；
- 同时报告 `count_match_rate`：`|P| == |E|` 的样本比例；
- 同时报告 signed count error：`|P| - |E|`。

**意义**：衡量“找得全不全、有没有多报”。

**单位**：比例（0–1）或计数。

**适用条件**：所有 `evaluation=direct` 的样本。

**误导性**：

- 查询词可能作为其他词的一部分出现，造成“真实但未列入 expected”的命中；规格预检必须捕获；
- `matched_text` 规范化可能导致跨行命中计数正确但文本形式不同；检测指标不受影响；
- 小样本上 macro 与 micro 可能差异很大，必须同时报告。

### 6.3 页码指标

不依赖 `char_start`，按页比较计数：

```text
page_count_exact_rate =
    (# pages where predicted_count(page) == expected_count(page))
    / (# pages with expected_count > 0)

page_confusion_pairs =
    [(p, q, deficit_on_p, surplus_on_q)]  # 仅报告，不合成单一分数
```

**意义**：检测跨页串位、漏页、重复页。

**单位**：比例（0–1）、计数对。

**适用条件**：多页样本（A03、C02、D02、F 组合等）。

**误导性**：

- 单页样本上恒为 1，不能代表页码能力；
- 同一页多个命中时，页级计数正确不代表每个命中位置正确；必须结合几何指标；
- 跨页候选对只是启发式，不是严格错误证明。

### 6.4 几何指标

对每个匹配的 `(R, B_layout)`：

```text
cx = (left + right) / 2
cy = (bottom + top) / 2
center_error_pt = sqrt((cx_R - cx_B)^2 + (cy_R - cy_B)^2)

IoU = area(R ∩ B) / area(R ∪ B)
inside_ratio = area(R ∩ B) / area(R)
coverage = area(R ∩ B) / area(B)
width_ratio = width_R / width_B
height_ratio = height_R / height_B
```

报告：

- `center_error_pt`：mean、median、p95、max；
- `inside_ratio`：mean、min、低于阈值的数量；
- `IoU`：mean、median、min（**报告项**）；
- `width_ratio` / `height_ratio`：mean、min、max；
- 逐边误差：`|left_R - left_B|`、`|bottom_R - bottom_B|`、`|right_R - right_B|`、`|top_R - top_B|`。

**意义**：判断矩形是否落在正确文字上，而不是仅仅“页对了”。

**单位**：points（误差）、比例（IoU/ratio）。

**适用条件**：

- `truth_level=layout-exact`：严格阈值；
- `truth_level=layout-band`：放宽阈值；
- `truth_level=detection-only`：不报告几何；
- `AMBIGUOUS_MATCH`：排除。

**待校准的初始阈值（设计参数，不是实测结论）**：

| 真值等级 | `center_error_pt` 门 | `inside_ratio` 门 | 说明 |
|---|---|---|---|
| `layout-exact`（base-14） | ≤ 1.0 | ≥ 0.999 | 单行、水平文本 |
| `layout-band`（CID/TTF） | ≤ 2.0 | ≥ 0.95 | 字体度量或墨迹范围较宽 |
| 旋转基准对比 | ≤ 0.05 | — | 与未旋转版本的同一绘制比较 |

> 上述阈值必须在 M0-06B 完成后、基于**实际运行报告**校准；在校准前只能作为“观察门”，不能对外宣称质量结论。

**误导性**：

- 布局框包含 side bearing、行距、字体上下沿，`IoU` 天然小于 1；**不得**用单一 IoU 阈值判定失败；
- 单字符查询的 side bearing 占比大，`width_ratio` 可能明显小于 1；
- CJK 的 em box 与墨迹范围差异更大；
- 中心正确但尺寸错误的情况必须通过 `width_ratio` / `height_ratio` 检出；
- 预测矩形完全落在布局框内，但落在错误字符上时，`inside_ratio` 可能仍然很高；因此 `center_error` 与逐边误差不可省略。

### 6.5 跨行矩形覆盖指标

```text
expected_lines = 该命中 expected parts 的数量
predicted_rects = 该命中 rects 的数量

rect_count_exact = (predicted_rects == expected_lines)
line_recall = (# expected lines with exactly one assigned rect) / expected_lines
line_precision = (# predicted rects assigned to an expected line) / predicted_rects
merged_box_detected = any(rect vertically overlaps >= 2 expected line bands)
bbox_union_max_abs_error =
    max(|bbox.left - union.left|, |bbox.bottom - union.bottom|,
        |bbox.right - union.right|, |bbox.top - union.top|)
```

**意义**：确保跨行证据没有被合并成一个跨行大框，也没有漏掉某一行。

**单位**：比例、计数、points。

**适用条件**：C 场景及所有跨行样本。

**通过标准（设计参数）**：

- `rect_count_exact = true`；
- `line_recall = 1.0`；
- `line_precision = 1.0`；
- `merged_box_detected = false`；
- `bbox_union_max_abs_error ≤ 1e-6`（结构不变量）。

**误导性**：

- 行数正确但某行矩形偏移，`line_recall` 仍可能为 1；必须同时报告逐行中心误差；
- 如果 PDFium 把两行合并为一个文本行，expected 本身可能不成立；预检必须确认换行边界；
- `bbox_union_max_abs_error` 是结构一致性，不是独立精度真值。

### 6.6 失败分类

| 代码 | 含义 | 判定方式 |
|---|---|---|
| `F01_NO_HIT` | expected > 0，predicted = 0 | 计数 |
| `F02_MISSING_OCCURRENCE` | 0 < predicted < expected | 计数 |
| `F03_EXTRA_HIT` | predicted > expected | 计数 + FP |
| `F04_WRONG_PAGE` | 页计数不匹配，且存在跨页候选 | 页计数 + 混淆对 |
| `F05_WRONG_POSITION` | `center_error_pt` > 门 | 几何 |
| `F06_WRONG_SIZE` | `width_ratio` / `height_ratio` 超出带 | 几何 |
| `F07_LINE_STRUCTURE` | 矩形数不匹配、合并框、行漏配 | 跨行 |
| `F08_CASE_MISMATCH` | `match_case` 行为与预期不符 | H 场景 |
| `F09_ENGINE_ERROR` | 异常、5xx、超时 | runner |
| `F10_INVALID_FIXTURE` | 预检失败 | 预检 |
| `F11_AMBIGUOUS_MATCH` | 匹配歧义 | matcher |
| `F12_UNSUPPORTED_EXPECTATION` | 规格要求了不支持的能力 | 规格校验 |

报告必须为每个失败给出：

- 样本 ID、查询、`match_case`；
- expected 与 predicted 的页码、矩形、计数；
- 失败代码；
- 可复现命令或参数；
- 是否计入质量分母（`F10` / `F12` 不计入）。

### 6.7 耗时指标

测量对象：`search_anchors(pdf_path, query, match_case=...)` 的单次调用。

```text
first_call_ms = 第一次调用耗时
warmup        = 1 次（丢弃）
repeats       = 5 次（默认）
median_ms, p95_ms, max_ms = 对 repeats 统计
ms_per_page   = median_ms / page_count
ms_per_mib    = median_ms / (file_size_bytes / 1048576)
```

记录：

- Python 版本、`pypdfium2` 版本、`reportlab` 版本、`pypdf` 版本、平台；
- 是否命中 OS 文件缓存（首次调用 vs 后续调用分开报告）；
- 数据集版本与规格哈希。

**意义**：发现明显退化或异常慢的样本。

**单位**：毫秒（ms）、ms/page、ms/MiB。

**适用条件**：同一台机器、同一次运行内比较。

**误导性**：

- `search_anchors()` 每次重新打开 PDF，耗时包含 open + text page 构建；这不等于“冷启动”；
- OS 文件缓存、CPU 频率、容器资源竞争会显著影响结果；
- 不同机器、不同 PDF 大小、不同页数的绝对耗时不可直接比较；
- **M0-06B 不设耗时通过/失败门**；只记录基线。M0-06C 可在同机同数据集上设置相对回归门（例如相对基线中位数不超过某个倍数），但必须明确标注机器与数据集版本。

---

## 7. 报告与产出

### 7.1 输出格式

每次运行输出：

1. **JSON 报告**：机器可读，包含 summary、samples、failures、environment。
2. **Markdown 报告**：人类可读，包含汇总表、分场景表、失败列表、复现命令。
3. **控制台摘要**：只打印关键计数与报告路径。

JSON 约束：

- 使用 `json.dump(..., allow_nan=False)`；
- 未定义的指标写 `null`，不写 `NaN` / `Infinity`；
- 所有比例保留足够精度，但报告层可格式化；
- 不写入服务器绝对路径、用户名、主机名；
- 包含 `schema_version`、`dataset_version`、`spec_sha256`、`generator_version`。

### 7.2 报告结构（示意）

```json
{
  "schema_version": 1,
  "dataset_version": "v0",
  "spec_sha256": "...",
  "environment": {
    "python": "3.11.x",
    "pypdfium2": "...",
    "reportlab": "...",
    "pypdf": "...",
    "platform": "..."
  },
  "summary": {
    "samples_total": 0,
    "samples_evaluated": 0,
    "samples_skipped": 0,
    "tp": 0,
    "fp": 0,
    "fn": 0,
    "precision": null,
    "recall": null,
    "f1": null,
    "count_match_rate": null,
    "page_count_exact_rate": null,
    "center_error_pt": {"mean": null, "median": null, "p95": null, "max": null},
    "inside_ratio_min": null,
    "iou": {"mean": null, "median": null, "min": null},
    "rect_count_exact_rate": null,
    "line_recall": null,
    "line_precision": null
  },
  "samples": [],
  "failures": []
}
```

> 上表中的 `0` 与 `null` 是 schema 占位，不是实测结果。实际报告必须由运行产生。

### 7.3 结果解释规则

- 报告必须区分 `direct` / `boundary` / `unsupported` 三类样本；
- `boundary` / `unsupported` 的失败不得计入质量结论；
- `F10_INVALID_FIXTURE` 必须单独列出，修复数据集后重跑；
- 任何质量结论必须同时给出样本数、数据集版本、环境指纹与阈值版本；
- 不得从单次运行外推生产性能。

---

## 8. 预期目录结构

```text
backend/
├── benchmarks/
│   ├── __init__.py
│   ├── README.md                  # M0-06B 运行说明（后续阶段创建）
│   ├── datasets/
│   │   └── v0/
│   │       ├── manifest.json      # 数据集版本、样本列表、规格哈希
│   │       ├── schema.json        # JSON Schema 或等价校验描述
│   │       ├── A01_single_hit.json
│   │       ├── A02_same_page_repeats.json
│   │       └── ...
│   ├── generate.py                # 从规格生成 PDF（reportlab + pypdf）
│   ├── truth.py                   # T-DET / T-LAYOUT 计算
│   ├── runner.py                  # 唯一允许 import rag_anchor 的适配器
│   ├── match.py                   # 预测与真值匹配
│   ├── metrics.py                 # 纯函数指标计算
│   ├── report.py                  # JSON / Markdown 报告
│   └── cli.py                     # validate / generate / run
├── tests/
│   └── benchmark/
│       ├── test_spec_schema.py
│       ├── test_spec_recount.py
│       ├── test_truth_independence.py
│       ├── test_generate.py
│       ├── test_match.py
│       ├── test_metrics.py
│       └── test_report_schema.py
└── .benchmark-data/               # 运行时生成，必须 gitignore
    ├── v0/
    │   ├── pdfs/
    │   └── reports/
```

说明：

- `benchmarks/` 是独立评测包，不进入生产 `src/rag_anchor/` 的运行时依赖；
- 从 `backend/` 目录执行 `uv run python -m benchmarks.cli ...`，利用当前工作目录导入；
- 不修改 `pyproject.toml` 的打包配置；M0-06B 不新增依赖；
- `backend/.benchmark-data/` 在 M0-06B 加入 `.gitignore`；
- 已生成的 PDF 与报告不提交 Git；只提交 JSON 规格、生成器、测试与文档。

---

## 9. 分阶段开发计划

### 9.1 M0-06B：最小可靠基线

**目标**：在现有 Python 测试基础设施上，建立可重复运行的服务层评测基线。

**范围**：

- 数据集：`v0-baseline`，22 个样本，覆盖 A、B、C、E、F、H；
- 真值：`T-DET` + `T-LAYOUT`；
- 运行层：直接调用 `search_anchors()`，不做 HTTP；
- 指标：precision、recall、F1、count match、per-page count、center error、inside ratio、IoU（报告）、rect count、line recall/precision；
- 报告：JSON + Markdown；
- 预检：`pypdf` 结构 + 文本存在性 + 规格自洽；
- 测试：规格校验、真值独立性、生成器、匹配器、指标、报告 schema；
- 失败分类：至少覆盖 F01–F07、F09–F12。

**非目标**：

- 不评 D、G、I；
- 不做 HTTP API 评测；
- 不做跨实现验证；
- 不做人工标注；
- 不做 CI；
- 不设硬性质量门或性能门；
- 不修改 API、源码、生产依赖。

**交付物**：

- `backend/benchmarks/` 包；
- `backend/benchmarks/datasets/v0/` 规格；
- `backend/tests/benchmark/` 测试；
- `backend/benchmarks/README.md`；
- `.gitignore` 增加 `backend/.benchmark-data/`。

**验收**：

- 所有非 `skipped` 样本通过预检；
- `validate`、`generate`、`run` 三条命令可复现执行；
- 生成的 PDF 与报告不在 Git 状态中；
- 原有后端测试与新增评测测试全部通过；
- ruff check / format / lock 检查通过；
- 真值独立性静态测试通过。

### 9.2 M0-06C：扩展场景与独立验证

**目标**：补齐 A–I 全部场景，加入边界与负向控制，提升真值可信度。

**范围**：

- 数据集：追加 D、G、I 共 6 个样本，总计 28 个；
- 追加边界样本：B04、H04；
- 指标：增加 page confusion pairs、failure taxonomy 汇总、跨行逐行误差、耗时统计；
- 独立验证：
  - 正式化 `pypdf` 预检；
  - 可选跨实现验证（先探测工具可用性）；
  - 每场景至少 1 个人工 spot-check；
- 运行层：可选增加 HTTP 传输适配器，用 `TestClient` 验证 API 序列化与错误映射；
- 质量门：只有在基线报告评审后，才引入 `--fail-under`；阈值必须记录版本与依据；
- 错误守护：增加缺失文件、损坏 PDF、空 query、非法 UUID 等非质量样本。

**非目标**：

- 不做 OCR 质量评测；
- 不做双栏阅读顺序正确性评测；
- 不做表格结构解析评测；
- 不做跨机器性能排名。

**验收**：

- 所有样本有明确分类：直接评测 / 边界 / 不支持；
- 不支持样本不被计入质量分母；
- 失败分类覆盖所有观察到的失败；
- 指标单元测试覆盖空集合、零预测、多余预测、跨页、跨行、歧义匹配；
- 报告可在无网络、无浏览器、无管理员权限的容器内生成。

### 9.3 M0-06D：稳定化与公开发布

**目标**：把评测体系变成可长期维护、可公开发布的数据集与工具。

**范围**：

- 冻结 `v1.0` 数据集；
- 数据集版本策略：任何规格或生成器语义变更都 bump 版本；
- 报告模板与结果解释指南；
- 复现指南：环境、命令、预期产物、已知限制；
- 可选 CI：在外部 CI 环境运行 `validate + run`，不要求本地 Docker CLI；
- 许可证与再分发审查：
  - 只发布规格与生成器，不发布 PDF；
  - 记录 PDFium / pypdfium2 / reportlab / pypdf 的许可信息；
  - 不重新分发系统字体文件；
- 性能基线表：记录一次同机运行的中位数与 p95，标注机器与版本，不作为跨机器门。

**验收**：

- 全新克隆可按文档重新生成并运行；
- 仓库中无二进制 PDF；
- 报告包含环境指纹与数据集哈希；
- 许可证审查记录完整；
- 数据集贡献规则与弃用策略写入文档。

---

## 10. 潜在风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 真值自我验证 | 指标虚高，无法发现真实缺陷 | T-DET/T-LAYOUT；禁止 PDFium 进入真值模块；静态导入测试 |
| 布局框 ≠ 墨迹框 | IoU 系统性偏低，误判失败 | 以 center error + inside ratio 为主；IoU 只报告；阈值校准 |
| 字体度量差异 | 布局框偏差 | base-14 严格；CID/TTF 用 `layout-band`；度量不可用则 skip |
| segment 化绘制改变词边界 | 查询不可检索或出现额外空格 | `pypdf` 预检；规格自洽扫描；必要时调整 segment |
| PDFium 文本规范化 | `matched_text` 与预期不同 | 不 gate `matched_text`；以计数、页码、几何为准 |
| `char_start` 索引空间不稳定 | 匹配错位 | 不把 `char_start` 当匹配键或真值 |
| 旋转页文本顺序变化 | 顺序断言误报 | 不 gate 顺序；只比较矩形与元数据 |
| 双栏/表格阅读顺序 | 期望错误 | 明确标记不支持；只评检测、页码、坐标 |
| 重叠匹配语义未承诺 | H04 误判 | 边界样本，只记录，不设门 |
| 非 BMP 字体缺失 | 样本无法生成 | 探测字体；缺失则 `skipped`，记录原因 |
| 非 BMP 查询不支持 | 误判为缺陷 | 标记 `unsupported`；只验证安全返回 0 |
| 扫描件负向控制被误解为 OCR 评测 | 对外表述错误 | 文档与报告明确“仅负向控制，不评 OCR 质量” |
| 运行时噪声 | 性能结论不可靠 | 多次重复、中位数/p95、环境指纹；不设绝对门 |
| 数据集漂移 | 历史结果不可比 | 版本号 + 规格 SHA-256 + 生成器版本 |
| 报告含 NaN/Infinity | JSON 不可消费 | `allow_nan=False`；未定义写 `null` |
| 生成 PDF 进入 Git | 仓库膨胀、版权风险 | `.benchmark-data/` gitignore；测试检查 |
| 字体再分发风险 | 许可问题 | 只引用运行时系统字体；不提交字体文件；发布前审查 |
| 容器无跨实现工具 | T-CROSS 无法执行 | 标记 `not_run`；不得写成 `passed` |
| 阈值过早固化 | 过拟合或误导 | M0-06B 只观察；M0-06C 校准并记录依据 |

---

## 11. 第一个最小实现版本的验收标准

以下标准针对 **M0-06B**，全部可二进制判定：

### 11.1 功能与结构

1. 存在 `backend/benchmarks/` 包，包含 `generate.py`、`truth.py`、`runner.py`、`match.py`、`metrics.py`、`report.py`、`cli.py`。
2. 存在 `backend/benchmarks/datasets/v0/manifest.json` 与至少 22 个样本规格，覆盖 A、B、C、E、F、H。
3. 每个样本包含：内容、查询词、期望命中数、期望页码、坐标真值来源、成功标准、可能失败原因；规格字段可被 schema 校验。
4. `backend/tests/benchmark/` 存在规格校验、真值独立性、生成器、匹配器、指标、报告测试。

### 11.2 命令与可复现性

在 `backend/` 目录执行以下命令：

```bash
UV_LINK_MODE=copy uv run python -m benchmarks.cli validate --dataset v0
UV_LINK_MODE=copy uv run python -m benchmarks.cli generate --dataset v0 --out .benchmark-data/v0
UV_LINK_MODE=copy uv run python -m benchmarks.cli run --dataset v0 --out .benchmark-data/v0 --format json,md
```

验收：

5. `validate` 退出码 0；无 `F10` / `F12`。
6. `generate` 退出码 0；生成的 PDF 页数、`/Rotate`、`/CropBox` 与规格一致。
7. `run` 退出码 0；生成 JSON 与 Markdown 报告；报告可被 `json.load` 解析，且不含 `NaN` / `Infinity`。
8. 重复执行 `run`，在相同环境下检测计数与页码计数一致；几何指标在浮点容差内一致。

### 11.3 真值独立性

9. `tests/benchmark/test_truth_independence.py` 通过：`generate.py`、`truth.py`、`match.py`、`metrics.py`、`report.py` 不导入 `pypdfium2` 或 `rag_anchor`。
10. 所有 expected 来自规格与 `reportlab` 度量；不存在从 `search_anchors()` / `get_charbox()` 复制 expected 的代码路径。
11. `pypdf` 预检覆盖所有非 `skipped` 样本；预检失败必须归类为 `F10_INVALID_FIXTURE`。
12. 规格自洽扫描通过：每个查询的 expected 数量等于规格文本中的独立字符串计数。

### 11.4 指标正确性

13. `tests/benchmark/test_metrics.py` 覆盖至少以下合成用例：
    - 完美匹配；
    - 漏检（F01/F02）；
    - 多报（F03）；
    - 页码错误（F04）；
    - 位置偏移（F05）；
    - 尺寸异常（F06）；
    - 跨行矩形数不匹配/合并（F07）；
    - 零 expected + 零 predicted；
    - 零 expected + 非零 predicted；
    - 歧义匹配（F11）。
14. 未定义指标输出 `null`，不输出 0 或 1。
15. `bbox == union(rects)` 的结构检查误差 ≤ `1e-6`。

### 11.5 质量门与报告

16. M0-06B **不设硬性质量门**；`run` 的退出码只反映 harness 是否成功，不反映系统质量。
17. 报告必须分别列出 `direct`、`boundary`、`unsupported` 样本；后两类不计入质量分母。
18. 报告必须列出所有失败及其分类码；`F10` / `F12` 单独列出。
19. 报告必须包含数据集版本、规格 SHA-256、生成器版本、Python/库版本与平台指纹。
20. 报告不得包含服务器绝对路径、用户名或主机名。

### 11.6 仓库与工具链

21. `backend/.benchmark-data/` 已在 `.gitignore` 中；`git status --short` 不显示生成的 PDF/报告。
22. 未修改 `backend/pyproject.toml` 与 `backend/uv.lock`；未新增依赖。
23. `UV_LINK_MODE=copy uv run pytest -q`：原有测试全部通过，新增 benchmark 测试全部通过。
24. `UV_LINK_MODE=copy uv run ruff check .`、`uv run ruff format --check .`、`uv lock --check` 通过。
25. `git diff --check` 退出码 0。
26. 不执行 `git add` / `git commit` / `git push`（除非后续任务明确要求）。

---

## 12. 关键取舍

| 取舍 | 选择 | 理由 |
|---|---|---|
| 服务层 vs HTTP 层 | M0-06B 先做服务层 | 复用现有 pytest；避免 HTTP/上传/存储噪声；坐标问题与传输层无关 |
| PDF 生成方式 | spec + reportlab 动态生成 | 可复现、可审查、无版权风险、无二进制 PDF |
| 查询段绘制 | segment 化，显式 x | 不依赖前缀宽度，布局真值更独立 |
| 真值来源 | T-DET + T-LAYOUT | 独立于 PDFium；足以为基线提供可解释真值 |
| 墨迹真值 | 不作为主真值 | 必须使用 PDFium，存在自我验证 |
| 几何主指标 | center error + inside ratio | 对布局框/墨迹框差异更稳健 |
| IoU | 只报告，不单独设门 | 布局框包含空白，IoU 天然小于 1 |
| 文本形式 | 不 gate `matched_text` | 引擎规范化不可控 |
| 字符索引 | 不 gate `char_start` | 历史已出现 PDFium/Python 索引不一致 |
| 匹配方式 | 同页最小费用最大匹配 | 先最大化匹配数；再最小化总中心距离；等代价与近邻歧义均标记 |
| 数据集格式 | JSON 规格 | 可版本化、可公开、可 schema 校验 |
| 报告产物 | gitignore | 避免二进制/环境相关结果污染仓库 |
| 性能 | 只记录，不设绝对门 | 容器环境噪声大；跨机器不可比 |
| 依赖 | 只用现有 `reportlab` / `pypdf` / pytest | 满足“避免不必要新依赖” |
| 扫描件 | 只做负向控制 | 当前无 OCR，不虚构能力 |
| 双栏/表格 | 只评检测与坐标 | 阅读顺序/结构解析未实现 |

---

## 13. 非目标

- 不做 OCR、扫描件文字识别质量评测。
- 不做前端 PDF.js 渲染、高亮、滚动居中的端到端评测。
- 不做 LLM、embedding、向量检索、rerank 评测。
- 不做加密 PDF、XFA、RTL、任意旋转文字的正确性评测。
- 不做生产级并发、吞吐、内存、长时间稳定性评测。
- 不做跨机器性能排名。
- 不做许可证法律结论。
- 不修改 API、源码、生产测试或依赖。

---

## 14. 参考文件

本设计基于以下已读取文件与已验收事实：

- `AGENTS.md`
- `README.md`
- `backend/README.md`
- `frontend/README.md`
- `docs/architecture.md`
- `docs/adr/0001-coordinate-contract.md`
- `backend/src/rag_anchor/services/anchor_service.py`
- `backend/src/rag_anchor/domain/models.py`
- `backend/src/rag_anchor/api/schemas.py`
- `backend/tests/unit/conftest.py`
- `backend/tests/unit/test_anchor_service.py`
- `backend/tests/unit/test_anchor_search_api.py`
- `backend/pyproject.toml`

坐标契约以 ADR-0001 与已验收实现为准：**absolute PDF user space、points、不减去 CropBox 原点、旋转不烘焙、页面尺寸为未旋转 CropBox 尺寸**。
