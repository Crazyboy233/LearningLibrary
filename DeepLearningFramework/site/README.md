# 深度学习框架学习手册 · 静态站点

把 `DeepLearningFramework/` 目录下的 Markdown 笔记汇总成的一个静态网页，
支持全文搜索、深浅色主题、公式排版与流程图渲染，**完全离线可用**。

## 打开方式

最简单的方式：直接双击 `index.html`（`file://` 协议下所有功能均可用）。

如果希望有更规范的本地服务（便于分享给同局域网的设备），可以在本目录执行：

```bash
python3 -m http.server 8934
# 然后访问 http://127.0.0.1:8934/
```

## 功能

| 功能 | 说明 |
| --- | --- |
| 全文搜索 | `⌘K` / `Ctrl+K` / `/` 唤起；支持多关键词与中文子串匹配，结果按相关度排序并高亮关键词片段 |
| 正文定位 | 点击搜索结果直接跳到对应章节，命中词在正文中高亮，并闪烁提示落点 |
| 深浅色主题 | 右上角切换，跟随系统偏好，选择写入 localStorage |
| 公式排版 | 219 个 LaTeX 公式在构建期用 KaTeX 渲染为静态 HTML，无需运行时 JS |
| 流程图 | 4 个 Mermaid 流程图按需加载渲染（仅在访问到含图的页面时加载 2.5 MB 运行时） |
| 图表查看 | 点开图片/流程图后可缩放（滚轮、触控板捏合、工具栏、双击）与拖动平移，打开时自动适应屏幕；页面内的宽图也能按住拖动平移 |
| 术语速查 | 「名词解释」页顶部提供术语索引 chips，一键跳转 |
| 折叠目录 | 右栏目录按标题层级折叠，默认只显示顶层；滚动到某章节会自动展开所在分支，也可手动逐个折叠或一键展开/收起 |
| 面板调宽 | 左右两栏宽度可拖拽（双击分隔线复位），选择记入 localStorage |
| 侧栏折叠 | 宽屏下左上角按钮可折叠 / 展开左侧导航，把空间让给正文；状态记入 localStorage |
| 移动端 | 单栏布局；左侧导航与右侧目录都是可滑出的抽屉，并适配刘海屏安全区 |
| 阅读辅助 | 顶部阅读进度条、回到顶部、图片点击放大、代码块一键复制 |

## 布局逻辑

| 区域 | 常驻条件 | 不满足时 |
| --- | --- | --- |
| 左侧导航 | 窗口 ≥ 1024px | 变成左侧抽屉，顶栏汉堡按钮唤起（宽屏下同一个按钮改为折叠/展开侧栏） |
| 右侧目录 | 窗口 ≥ 左栏宽 + 右栏宽 + 660px | 变成右侧抽屉，顶栏目录按钮唤起 |

右栏的判定是**动态**的：把左栏拖宽后，若剩余空间不够放正文，右栏会自动收成抽屉。
两栏宽度范围分别为 200–460px 与 170–420px。

## 目录结构

```
site/
├── index.html                  # 站点外壳（唯一入口）
├── setup.sh                    # 依赖检查 / 安装（Ubuntu，可反复执行）
├── README.md
├── .gitignore
├── tools/                      # 构建工具（不叫 build，避免被通用 .gitignore 规则吃掉）
│   ├── build.py                # 主构建：Markdown → HTML + 搜索索引
│   ├── render_math.cjs         # 构建期 KaTeX 服务端渲染
│   ├── smoke_test.cjs          # 无头逻辑自检（可选，需 jsdom）
│   └── vendor/katex.min.js     # 供 render_math.cjs 调用
└── assets/
    ├── css/style.css           # 全部样式与主题变量
    ├── js/app.js               # 路由 / 搜索 / 主题 / 交互
    ├── js/content.js           # ← 构建产物：页面 HTML + 搜索索引
    ├── img/                    # ← 构建产物：从源笔记复制过来的配图
    └── vendor/
        ├── katex/              # KaTeX 样式与字体（仅 woff2）
        └── mermaid.min.js      # 流程图渲染（懒加载）
```

## 环境准备

首次构建前先跑一次依赖脚本（面向 Ubuntu / Debian）：

```bash
./setup.sh            # 检查并安装缺失的依赖
./setup.sh --check    # 只检查环境，不做任何安装
```

它会依次确认 Python 3、`python3-venv`、`markdown` / `pygments`、Node.js（用于渲染公式），
**已满足的项自动跳过**，可以反复执行。脚本只准备依赖，不会触发构建。

- 系统 Python 已具备 `markdown` + `pygments` → 直接使用它
- 否则在 `site/` 下创建 `.venv` 并安装（Ubuntu 24.04 起系统 Python 受 PEP 668 保护，
  venv 是更稳妥的做法）

## 重新构建

笔记内容更新后，在 `site/` 目录下执行（`setup.sh` 结束时会打印你当前环境对应的那条）：

```bash
.venv/bin/python tools/build.py    # 脚本创建了虚拟环境时
python3 tools/build.py             # 系统 Python 已具备依赖时
```

脚本会重新读取三个源文档并覆盖 `assets/js/content.js`：

- `../名词解释.md` → 框架（Ray / TensorFlow / PyTorch）、名词解释、并行训练策略、训练流程、激活函数、损失函数、Transformer 入门、MLP
- `../Transformer.md` → Attention Is All You Need（论文精译）
- `../训练流程图.md` → PS 架构与 Worker 计算流程

依赖：`markdown`、`pygments`（Python）与 `node`（用于 KaTeX 渲染）。

```bash
pip install markdown pygments
```

## 部署到服务器

整站是纯静态的，把 **`index.html` + `assets/`** 传到服务器任意目录即可，
`tools/`、`setup.sh` 与 `README.md` 不需要上线。

- **零配置**：路由走 URL hash（`#/glossary`），服务器不需要任何 rewrite 规则
- **支持子目录**：所有资源都是相对路径，放在 `https://example.com/notes/` 下也能直接用
- **无外部依赖**：KaTeX 字体、Mermaid、图标全部在本地，不请求任何 CDN

```bash
# 只同步运行需要的文件
rsync -av --exclude 'tools' --exclude 'setup.sh' --exclude 'README.md' site/ user@host:/var/www/notes/
```

### 更新内容时

重新跑一次构建（见上一节）后，**通常只需重新上传 `assets/js/content.js`**
（有新增配图时再传 `assets/img/`）。

如果服务器给 `.js` 配了长缓存，建议对 `content.js` 设 `Cache-Control: no-cache`，
否则访客可能一直看到旧内容，例如 Nginx：

```nginx
location ~* /assets/js/content\.js$ {
    add_header Cache-Control "no-cache";
}
```

## 页面切分规则

页面划分写在 `tools/build.py` 顶部的 `PAGES` 列表里：每项通过 `start` / `end`
两个「原文标题行」圈定内容范围。要新增或调整章节，改这个列表即可，
其余（导航、搜索索引、上下页、目录）都会自动生成。
