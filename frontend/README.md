# RAG-Anchor Frontend

`PDF Evidence Explorer` 前端：上传 PDF、检索证据文字，并用 PDF.js 在原文上精确高亮所有命中位置；点击命中后自动定位并居中。

技术栈：Vite 5 + TypeScript 5 + 原生 HTML/CSS + `pdfjs-dist` 4 + Vitest 2。
不使用 React / Vue / UI 组件库 / 状态管理库。

## 环境要求

- Node.js **18.20.8** 或更高（`engines.node` 为 `>=18.0.0`）
- npm 10.x
- 后端运行在 `http://127.0.0.1:8000`

## 安装与启动

```bash
cd frontend
npm install
npm run dev
```

开发服务器监听 `0.0.0.0:5173`，浏览器访问 `http://<host>:5173`。

开发服务器把 `/api` 代理到 `http://127.0.0.1:8000`，因此不需要后端开启 CORS。

## 脚本

| 命令                | 说明                                |
| ------------------- | ----------------------------------- |
| `npm run dev`       | 启动 Vite 开发服务器                |
| `npm run typecheck` | `tsc --noEmit`（应用与 Vite 配置）  |
| `npm test`          | Vitest 单测（`tests/**/*.test.ts`） |
| `npm run build`     | 类型检查 + 生产构建到 `dist/`       |
| `npm run preview`   | 预览生产构建                        |

## 代理配置

默认代理目标为 `http://127.0.0.1:8000`。如需修改：

```bash
cp .env.example .env.local
# 编辑 .env.local
VITE_API_PROXY_TARGET=http://127.0.0.1:9000
```

`vite.config.ts` 使用 `loadEnv()` 读取 `.env`、`.env.local` 与 `.env.<mode>[.local]`，修改后需重启开发服务器。

## 目录结构

```text
frontend/
├── index.html               # 静态布局与全部 DOM 锚点
├── src/
│   ├── main.ts              # UI 控制器：上传/列表/搜索/命中导航/删除
│   ├── styles.css           # 设计令牌、布局、状态样式、响应式
│   ├── api/
│   │   ├── types.ts         # FastAPI 响应模型的 TypeScript 镜像
│   │   └── client.ts        # HTTP 请求、FormData、错误映射
│   ├── pdf/
│   │   └── viewer.ts        # PDF.js 加载、渲染、翻页、缩放、销毁
│   ├── highlight/
│   │   └── renderer.ts      # viewport 坐标转换与高亮 overlay
│   └── ui/
│       └── scroll.ts        # 命中居中滚动几何与渲染时序协调
└── tests/
    ├── api-client.test.ts
    ├── highlight-renderer.test.ts
    └── scroll.test.ts
```

## 关键实现说明

### PDF 加载

`GET /api/v1/documents/{document_id}/file` → `ArrayBuffer` → PDF.js `getDocument({ data })`。

M0-05B 不依赖 HTTP Range，每次选择文档都会完整加载 PDF（后端上传上限 20 MiB）。
Worker 通过 Vite `?url` 导入 `pdfjs-dist/build/pdf.worker.min.mjs`，与 `pdfjs-dist` 版本一致。

### 高亮坐标

后端返回的 `rects` 是 **absolute PDF user space**（points），遵循
[ADR-0001](../docs/adr/0001-coordinate-contract.md)。

前端只做一次映射：

```text
PDF user space → page.getViewport({ scale, rotation }) → viewport.convertToViewportRectangle()
→ min/max 归一化 → 绝对定位的 overlay div
```

不做 CropBox 平移，不重复应用页面旋转，不把 points 当 CSS 像素。
多行证据按后端 `rects` 逐行绘制，不合并成跨行大框。

### 命中自动居中

点击任意命中项后，`activateHit()` 先切换到命中所在页，等待页面渲染与高亮绘制完成，再把 active 高亮滚动到 `#viewer-stage` 视口中央。同页与跨页使用同一条流程；连续快速点击时，旧的滚动请求会被丢弃，不会覆盖最新命中。

缩放或窗口尺寸变化会触发 PDF.js 重新渲染并调用 `onRender`，高亮层随即用新的 viewport 重新映射所有 `rects`。前端不重新计算字符坐标或后端 `bbox`，只做 PDF user space → 屏幕坐标的 viewport 转换。

### 状态与竞态

- 切换文档、翻页、缩放都会递增渲染 token，过期异步结果不会覆盖最新状态。
- `RenderingCancelledException` 静默忽略。
- 切换文档会清除旧高亮；重新搜索会清除旧结果。
- 删除当前文档会先关闭 PDF 阅读器再刷新列表。

## 测试范围

- API client：列表、上传字段名 `file`、搜索请求体、零命中、错误 `detail`、网络错误、文件字节、204 删除。
- 高亮转换：普通矩形、轴反转、缩放比例、非默认 CropBox 原点、多行矩形不合并、零面积过滤、active 标记。
- 命中居中：同页/跨页定位、上下方向、连续点击竞态、边界夹紧、多行命中并集、结果列表最小滚动。

`paintHighlights()` 依赖 DOM，当前 Vitest 使用 `environment: "node"`，因此只测试纯函数部分。

## 已知限制

- 本地开发 MVP：无认证、无速率限制、无 OCR。
- 单页渲染（只渲染当前页），不做连续滚动或虚拟化；命中自动居中只滚动阅读器视口。
- 大文件需全量加载到内存（上限 20 MiB）。
- 不支持扫描件、加密 PDF、任意角度旋转文字、RTL、复杂表格阅读顺序。
- 尚未进行真实浏览器自动化测试；worker 与真实 PDF 渲染需手动验证。
