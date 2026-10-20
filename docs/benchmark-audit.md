# M0-06C-Lite Benchmark Reproducibility Audit

> 审查类型：可复现性审查 + 独立证据抽检（Lite）
> 记录日期：2026-10-20
> 审查人：RAG-Anchor 评测质量工程师（AI 助理）
> 结论速览：**v0 基线可复现，抽检证据一致，允许进入 M1-01**（详见 F 节）

---

## A. 基线信息

| 项目 | 值 |
|---|---|
| Git commit | `ad636c5`（`test: complete 22-sample PDF anchor benchmark baseline`） |
| 运行时间（report `generated_at_utc`） | `2026-10-20T08:28:34Z` |
| 报告来源 | `backend/.benchmark-data/v0/report.json` |
| 数据集 | `v0`（`dataset_sha256 = 59cc7a031b10c8127900a2747d0215a85662d6f9f19e01248185457e658b1e00`） |
| 运行环境 | Python 3.11.13 / uv；pypdfium2 5.14.0、reportlab 5.0.1、pypdf 6.21.1 |

复现命令（在 `backend/` 下执行）：

```bash
UV_LINK_MODE=copy uv run python -m benchmarks.cli validate --dataset v0
UV_LINK_MODE=copy uv run python -m benchmarks.cli generate --dataset v0
UV_LINK_MODE=copy uv run python -m benchmarks.cli run --dataset v0 --format json,md
```

三个子命令退出码均为 `0`。

## B. 运行结果

以下数值**逐字段取自本次运行的 `report.json`**，未做任何修改：

| 指标 | 值 |
|---|---:|
| 样本总数 | 22 |
| `direct` | 19 |
| `boundary` | 2（B03、H04） |
| `unsupported` | 1（B04） |
| `quality_sample_count` / `excluded_sample_count` | 19 / 3 |
| `query_count` | 21 |
| TP / FP / FN | 33 / 0 / 0 |
| precision / recall / F1（微平均与宏平均） | 1.0 / 1.0 / 1.0 |
| 平均中心误差 `center_error_pt.mean` | 0.35179649 pt |
| 平均 IoU `iou.mean` | 0.93956088 |
| 平均包含比 `inside_ratio.mean` | 0.99553932 |
| 页码统计 `pages_with_expected` / `exact_pages` | 23 / 23 |
| `page_count_exact_rate` | 1.0 |
| 歧义配对排除数 | 0 |
| 失败记录 `failures` | `{}`（无任何失败码） |
| `rect_count_exact_rate` / `line_recall` / `line_precision` | 1.0 / 1.0 / 1.0 |
| `merged_box_count` | 0 |

## C. 抽检样本（A03 / E02 / F04 / B01）

抽检方法：用**独立于 PDFium** 的真值与工具核对生产服务输出。逐样本证据：

### A03（多页）

- 规格：3 页，每页一行 `narwhal` 语句。
- 独立计数（T-DET）：3。
- 生产服务：3 个命中，`page_index = [0, 1, 2]`，每页恰好 1 个。
- 注意区分：**真实页码是 1/2/3**（`page_number`），而每页命中数量分布是 `[1, 1, 1]`；两者口径不同，报告中分别记录。
- 中心误差约 1.158 pt，在布局框口径下正常。

### E02（非默认 CropBox + 跨行）

- 规格：CropBox `[70, 120, 450, 560]`，两行 `alpha beta` / `gamma delta`，查询 `beta gamma`（跨行）。
- 独立计数：1；生产服务返回 1 个命中，含**两个行矩形**（每行一个），未被合并成单个大框。
- 关键契约验证：命中矩形坐标**没有减去 CropBox 原点**（如 `hit_bbox.left ≈ 100.48`，而非 `100.48 − 70`），符合 ADR-0001 的绝对 PDF 用户空间契约。
- 中心误差约 0.111 pt。

### F04（旋转 + CropBox 组合）

- 规格：CropBox `[50, 100, 400, 500]`，`/Rotate = 90`，查询 `zephyr`。
- 生产服务返回的矩形仍处于**未旋转的绝对 PDF 用户空间**（x ≈ 124），旋转仅通过 `PageDescriptor.rotation = 90` 单独表达，未烘焙进矩形。
- `width_pt` / `height_pt` 为未旋转 CropBox 尺寸（350.0 / 400.0）。
- 中心误差约 0.184 pt。

### B01（中文前缀 + ASCII 查询）

- 规格：单行 `中文 quick 证据`（WQY MicroHei 字体），benchmark 声明查询为 `quick`。
- PDFium 与 Poppler（`pdftotext`）**两个引擎**都能从该 PDF 提取出中文内容，中文字符存在且可读。
- 独立计数：1；生产服务：1 个命中，`matched_text == "quick"`。
- 中心误差约 0.286 pt。

## D. 独立性说明

本次抽检中各证据来源的角色：

| 来源 | 角色 | 说明 |
|---|---|---|
| **T-DET** | 独立真值 | 由规格文本独立计算命中数量，与任何 PDF 引擎无关 |
| **T-LAYOUT** | 独立真值 | 由 ReportLab 绘制参数（原点 + 字体度量）计算几何真值，独立于 PDFium |
| **pypdf** | 独立核验 | 独立检查 PDF 页数与页面元数据（CropBox / Rotate） |
| **Poppler `pdftotext`** | 独立核验 | 辅助验证**跨引擎文本提取**（文本内容层面） |
| **PDFium** | **被测对象** | 生产服务的输出，**不是预期真值** |

关于跨引擎核验范围的**如实声明**：

- 本次**没有**展示经过坐标系转换后的 Poppler 字符框逐项对照。
- 因此**不能声称**已完成第二个渲染引擎的逐字符几何坐标交叉验证。
- 本次**已完成**的是：跨引擎**文本提取**核验（Poppler `pdftotext`）、PDF 结构核验（pypdf 页数与页面元数据）、独立布局真值（T-LAYOUT）与生产坐标（PDFium）的比对。

## E. 已知限制

1. 这是**合成 PDF 回归基线**，不是通用真实 PDF 质量排行榜；`Precision = 1.0` 仅在该 22 样本数据集内成立。
2. B03 / B04 的真正非 BMP 字符 U+1D11E（`𝄞`）存在 **ToUnicode 编码损坏**问题（reportlab 5.0.1 写出奇数长度十六进制串，PDFium 将其截断为 U+1D11 `ᴑ`）。这两个样本**不能**用于证明生产服务对非 BMP 查询的支持或不支持，仅记录能力边界。
3. 字体依赖**特定系统路径与版本**（`/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`、`/usr/share/fonts/truetype/freefont/FreeSerif.ttf`），仓库不包含字体文件；换环境可能影响几何真值。
4. A03、E02 等样本较低的 IoU 可能来自 **layout box（布局框）与实际字符墨迹框定义的差异**，属口径差异而非缺陷；`IoU` 仅作参考，不作通过门限。
5. **OCR / 扫描件与双栏阅读顺序**尚未完成支持和评测。

## F. 结论

- 当前 `v0` Benchmark 可以作为**可复现的工程回归基线**：CLI 三步全部退出 `0`，指标与设计文档记录一致，抽检证据与独立真值一致，工作区干净、评测产物未入 Git。
- **允许进入 M1-01：可追溯 PDF 文本抽取与 Chunk 切分**。
- 但**不宣称**所有未验证功能已经完成：非 BMP 查询能力、OCR/扫描件、双栏阅读顺序、第二个渲染引擎的逐字符几何交叉验证等仍属未验证范围（见 E 节）。

---

相关文档：

- 评测设计：[`docs/benchmark-plan.md`](benchmark-plan.md)
- 运行指南：[`backend/benchmarks/README.md`](../backend/benchmarks/README.md)
- 坐标契约：[`docs/adr/0001-coordinate-contract.md`](adr/0001-coordinate-contract.md)
