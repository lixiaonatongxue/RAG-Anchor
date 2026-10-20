# PDF 锚定评测（Benchmark）运行指南

本目录包含 RAG-Anchor 的**确定性 PDF 证据锚定评测（benchmark）**：22 个合成样本、独立真值、
以及可复现的 CLI 运行入口。

本指南面向未来的开源使用者与贡献者，所有命令都可以直接复制执行。

---

## 1. Benchmark 的用途

该 benchmark 用来验证一件事：

> 给定一个 PDF 和一段证据查询文本，生产锚定服务返回的
> **页码 + PDF 用户空间几何矩形**是否与独立计算的真值一致。

它评测的是「检出（detection）」与「几何（geometry）」两类结果：

- **检出**：命中数量、所在页码是否正确（TP / FP / FN）。
- **几何**：矩形位置是否落在真值附近（中心误差、IoU、包含比）。

评测使用**合成 PDF**（由 `reportlab` 按规格脚本生成），因此完全确定性、可重复，
不依赖任何真实用户文档。

## 2. 与生产搜索服务的关系

- Benchmark 直接调用生产锚定函数 `search_anchors()`
  （`backend/src/rag_anchor/services/anchor_service.py`），**不经过 HTTP**。
- `benchmarks/runner.py` 是 benchmark 中**唯一**允许导入 `rag_anchor` / `pypdfium2` 的模块；
  其余模块（`generate.py` / `truth.py` / `match.py` / `metrics.py` / `report.py` / `fonts.py`）
  被禁止导入生产代码，以保证真值独立。
- 因此 benchmark 衡量的是生产锚定路径本身，而不是某个测试替身。

坐标契约见 `docs/adr/0001-coordinate-contract.md`：矩形为 **PDFium 绝对 PDF 用户空间**点坐标，
不减去 CropBox 原点，旋转只记录在 `PageDescriptor.rotation` 中，不烘焙进矩形。

## 3. 样本分类（22 个）

数据集 `v0` 共 **22** 个样本，分为三类：

| 分类 | 数量 | 样本 | 含义 |
|---|---:|---|---|
| `direct` | **19** | A01 H01 C01 B02 A05 E01 E02 F01 F02 F03 A03 F04 H03 B01 C02 C03 A02 A04 H02 | 有独立真值，参与聚合质量指标 |
| `boundary` | **2** | B03 H04 | 边界行为样本，仅记录实际行为，**不计入**聚合质量分母 |
| `unsupported` | **1** | B04 | 当前未承诺支持的能力，仅记录 |

- `boundary` / `unsupported` 样本仍会运行并出现在逐样本报告中，
  但被聚合指标过滤（见 `benchmarks/metrics.py::_quality_records()`）。
- 因此聚合质量指标的 `quality_sample_count = 19`，`excluded_sample_count = 3`。

样本清单以 `backend/benchmarks/datasets/v0/manifest.json` 为准。
`truth.load_specs()` **只**读取 manifest 中列出的文件；磁盘上新增的规格文件在更新 manifest 前不会被加载。

## 4. 环境要求

- **Python 3.11**（`pyproject.toml` 要求 `>=3.11,<3.12`）。
- **uv** 作为包管理器与运行器（`uv.lock` 已提交，请勿忽略）。
- 依赖：`pypdfium2`（生产锚定）、`reportlab`（生成样本）、`pypdf`（写副本 / 结构校验）、
  `pytest`、`ruff`。

推荐设置 `UV_LINK_MODE=copy`，以避免在某些文件系统（如 overlayfs）上 uv 硬链接失败。

## 5. 运行目录与完整命令

所有命令都在 **`backend/`** 目录下执行（该目录包含 `benchmarks/` 与 `tests/`）：

```bash
# 从仓库根目录进入后端
cd backend

# 1) 校验样本规格（不生成 PDF）
UV_LINK_MODE=copy uv run python -m benchmarks.cli validate --dataset v0

# 2) 生成确定性样本 PDF 并做 preflight 检查
UV_LINK_MODE=copy uv run python -m benchmarks.cli generate --dataset v0

# 3) 运行评测并写出 JSON / Markdown 报告
UV_LINK_MODE=copy uv run python -m benchmarks.cli run --dataset v0 --format json,md

# 4) 运行 benchmark 单元测试
UV_LINK_MODE=copy uv run pytest tests/benchmark -q -o addopts=""
```

三个 CLI 子命令的退出码反映的是**评测框架是否正常完成**，而不是锚定质量高低：

- `validate`：读取并校验样本规格，退出码 `0` 表示规格全部合法。
- `generate`：生成 PDF 并跑 preflight，退出码 `0` 表示每个 fixture 都通过结构 + 文本预检。
- `run`：生成 fixture → 调用生产服务 → 与独立真值匹配 → 写报告，退出码 `0` 表示流程跑通。

> 可选静态检查（在 `backend/` 下）：

```bash
UV_LINK_MODE=copy uv run pytest -q -o addopts=""
UV_LINK_MODE=copy uv run ruff check .
UV_LINK_MODE=copy uv run ruff format --check .
UV_LINK_MODE=copy uv lock --check
```

## 6. 系统字体依赖

样本生成依赖宿主系统已安装的字体，**字体文件不随仓库提交**。
字体注册代码见 `backend/benchmarks/fonts.py`：

| 字体 | 绝对系统路径 | 依赖样本 | 说明 |
|---|---|---|---|
| `WQYMicro` | `/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`（subfont 0） | B01 | 中英混排 |
| `FreeSerif` | `/usr/share/fonts/truetype/freefont/FreeSerif.ttf` | B03、B04 | 非 BMP 字符样本 |
| `Helvetica` | reportlab 内置 | 其余 19 个样本 | 始终可用 |

关键约束：

- 当前实现使用**绝对系统路径**引用字体。
- 字体缺失或无法被 reportlab 解析时，代码会**显式报错**
  （生成阶段 `GenerationError`，真值阶段 `TruthError`），**不会静默回退**到 `.notdef` 字形。
- 不同字体**版本**可能改变字形宽度/度量，从而影响几何真值。
- 仓库**不包含**任何字体文件。

因此，**不能声称该数据集在所有 Linux 环境上完全开箱即用**；
请在具备上述字体的主机上运行，或在缺字体时按报错提示安装对应系统字体。

## 7. 报告输出位置

评测产物位于：

```text
backend/.benchmark-data/v0/
├── report.json     # 结构化报告（机器可读）
├── report.md       # 人类可读报告
└── pdfs/           # 生成的 22 个样本 PDF
```

该目录**不应提交 Git**（已在 `.gitignore` 中忽略：`backend/.benchmark-data/`）。

## 8. 如何解读指标

报告 `summary` 分四组。核心字段：

**检出（`summary.detection`）**

| 字段 | 含义 |
|---|---|
| `tp` / `fp` / `fn` | 命中配对后的真阳 / 假阳 / 假阴 |
| `precision` / `recall` / `f1` | 微平均检出指标 |
| `macro` | 宏平均（按样本平均） |
| `n_defined` | 参与该指标计算的样本数（分母为 0 的样本计为 `null`） |
| `count_match_rate` | 每个查询的命中数量是否与真值完全相等的比例 |

**页码（`summary.page`）**

| 字段 | 含义 |
|---|---|
| `pages_with_expected` / `exact_pages` | 有期望命中的页数 / 页分布完全正确的页数 |
| `page_count_exact_rate` | 页分布完全正确比例 |
| `confusion_pairs` | 页号混淆对（若有） |

**几何（`summary.geometry.occurrence`）**

| 字段 | 含义 |
|---|---|
| `center_error_pt` | 预测矩形中心与布局真值中心的距离（点） |
| `iou` | 预测矩形与布局真值框的 IoU |
| `inside_ratio` | 预测矩形落在真值框内的比例 |
| `coverage` | 真值框被预测矩形覆盖的比例 |

**多行（`summary.multiline`）**

| 字段 | 含义 |
|---|---|
| `matched_occurrences` | 参与多行统计的命中数 |
| `rect_count_exact_rate` | 每命中行矩形个数正确的比例 |
| `line_recall` / `line_precision` | 逐行召回 / 精确率 |
| `merged_box_count` | 被错误合并成单个框的命中数（应为 0） |

关于几何真值的口径：

- 真值 `T-LAYOUT` 是**布局框**（由绘制原点 + reportlab 字体度量推导），
  不是逐字符墨迹框。因此 `IoU < 1` 属正常现象，`IoU` **仅作参考，不作为通过门限**。
- `center_error_pt` 是更稳定的几何指标。
- 被判定为「歧义最近邻」的配对被排除出几何统计，但仍计入检出（TP/FP/FN）。

## 9. 为什么 `Precision = 1.0` 不代表任意真实 PDF 100% 准确

报告中的 `precision = 1.0`、`recall = 1.0` 只说明：

> 在 **22 个合成样本、19 个 direct 样本**上，生产锚定结果与独立真值完全一致。

它**不能**外推为「任意真实 PDF 都 100% 正确」，原因包括：

- 样本是**合成 PDF 回归基线**，不是通用真实 PDF 质量排行榜；
- 未覆盖 OCR / 扫描件、双栏阅读顺序、复杂排版等真实场景；
- 非 BMP 字符（如 U+1D11E）存在 ToUnicode 编码问题，B03/B04 只记录能力边界；
- 几何真值是布局框口径，`IoU` 天然小于 1；
- 字体依赖特定系统路径与版本，换环境可能改变几何真值。

请把该 benchmark 当作**工程回归基线**使用：它防止回归，不宣告通用准确率。

## 10. 常见问题与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'benchmarks'` | 不在 `backend/` 目录下运行 | `cd backend` 后再执行 |
| `GenerationError` / 字体缺失报错 | 缺少 `wqy-microhei.ttc` 或 `FreeSerif.ttf` | 安装对应系统字体，或换到具备字体的主机 |
| `TTFError ... postscript outlines are not supported` | 使用了 CFF/PostScript 轮廓字体 | 使用 `fonts.py` 中登记的 TrueType 字体 |
| uv 报硬链接错误 | overlayfs 等文件系统 | 使用 `UV_LINK_MODE=copy` |
| `validate` 报 `ambiguous centers` | 同页同查询的命中中心间距小于 `min_separation_pt` | 调整样本 `min_separation_pt`，或修正文本布局 |
| `preflight` 失败 | fixture 结构或提取文本与规格不符 | 查看 `generate` 的 `[FAIL]` 明细 |
| B03 / B04 结果与直觉不符 | 非 BMP ToUnicode 编码损坏（详见 `docs/benchmark-plan.md`） | 属已知能力边界，不作为缺陷 |
| 想了解完整设计与真值模型 | — | 阅读 `docs/benchmark-plan.md` |
| 想了解复现性审查结论 | — | 阅读 `docs/benchmark-audit.md` |

---

相关文档：

- 设计文档：[`docs/benchmark-plan.md`](../../docs/benchmark-plan.md)
- 复现性审查记录：[`docs/benchmark-audit.md`](../../docs/benchmark-audit.md)
- 坐标契约：[`docs/adr/0001-coordinate-contract.md`](../../docs/adr/0001-coordinate-contract.md)
- 生产锚定服务：`backend/src/rag_anchor/services/anchor_service.py`
