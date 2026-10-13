#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLearningFramework 学习笔记静态站点 —— 构建脚本

作用:
  1. 读取目录下的 Markdown 源文档, 按章节切分并渲染为 HTML
  2. 构建期渲染 LaTeX 公式 (KaTeX 服务端渲染) 与代码高亮 (Pygments)
  3. 生成全文搜索索引
  4. 输出 site/assets/js/content.js

用法:
  python3 build.py
依赖:
  markdown, pygments  (pip install markdown pygments)
  node  (用于 KaTeX 服务端渲染, 依赖 build/vendor/katex.min.js)
"""

from __future__ import annotations

import html as htmllib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import markdown
from markdown.treeprocessors import Treeprocessor
from pygments import highlight as pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

BUILD_DIR = Path(__file__).resolve().parent
SITE_DIR = BUILD_DIR.parent
ROOT_DIR = SITE_DIR.parent

OUT_JS = SITE_DIR / "assets" / "js" / "content.js"
IMG_DIR = SITE_DIR / "assets" / "img"

NODE_BIN = shutil.which("node") or "/opt/homebrew/bin/node"

# --------------------------------------------------------------------------
# 页面定义: 源文档 -> 站点页面
#   start / end 为原文中精确匹配的标题行, 二者之间(不含 start 行)为页面内容
# --------------------------------------------------------------------------
PAGES = [
    dict(id="ray", group="深度学习框架", title="Ray",
         desc="Python 原生统一分布式计算框架，一套 API 覆盖数据处理、训练、调优、推理与强化学习",
         badge="分布式", src="名词解释.md", start="## Ray", end="## TensorFlow"),
    dict(id="tensorflow", group="深度学习框架", title="TensorFlow",
         desc="从张量、计算图到 Serving / Lite / JS 的全链路框架组成",
         badge="框架", src="名词解释.md", start="## TensorFlow", end="## Pytorch"),
    dict(id="pytorch", group="深度学习框架", title="PyTorch",
         desc="动态图优先的深度学习框架，本节记录其学习入口与资料索引",
         badge="框架", src="名词解释.md", start="## Pytorch", end="# 名词解释"),

    dict(id="glossary", group="核心术语", title="名词解释",
         desc="深度学习与分布式训练高频术语速查：从 Attention、AUC 到 ZeRO、残差",
         badge="术语速查", src="名词解释.md", start="# 名词解释", end="# 并行训练策略"),
    dict(id="parallel", group="核心术语", title="并行训练策略",
         desc="DDP 数据并行、PP 流水线并行、TP 张量并行的分工与差异",
         badge="分布式", src="名词解释.md", start="# 并行训练策略", end="# 训练流程"),
    dict(id="activations", group="核心术语", title="激活函数",
         desc="Sigmoid / Tanh / ReLU / Leaky ReLU / Softmax 的定义与取舍",
         badge="基础", src="名词解释.md", start="# 激活函数", end="# 损失函数"),
    dict(id="losses", group="核心术语", title="损失函数",
         desc="均方误差、交叉熵、BCEWithLogitsLoss 的适用场景",
         badge="基础", src="名词解释.md", start="# 损失函数", end="# Transformer"),
    dict(id="mlp", group="核心术语", title="最基础的神经网络 MLP",
         desc="多层感知机：一切深度网络的起点",
         badge="基础", src="名词解释.md", start="# 最基础的神经网络MLP", end=None),

    dict(id="training-loop", group="训练全流程", title="训练流程",
         desc="从离线特征工程、在线特征处理到 forward / backward / update 的完整链路",
         badge="流程", src="名词解释.md", start="# 训练流程", end="# 激活函数"),
    dict(id="training-flow", group="训练全流程", title="训练流程图 · PS 架构",
         desc="参数服务器与 Worker 的 Pull / Push 协同，以及单节点前向反向计算的每一步",
         badge="流程图", src="训练流程图.md", start=None, end=None),

    dict(id="transformer-intro", group="Transformer", title="Transformer 入门",
         desc="核心思想、编码器解码器两大组成部分与关键模块速览",
         badge="架构", src="名词解释.md", start="# Transformer", end="# 最基础的神经网络MLP"),
    dict(id="paper-transformer", group="Transformer", title="Attention Is All You Need",
         desc="《Attention Is All You Need》全文精译，含注意力公式、复杂度对比与消融实验",
         badge="论文精读", src="Transformer.md",
         start="# Attention Is All You Need", end=None),
]

GROUP_DESC = {
    "深度学习框架": "主流框架的定位、分层架构与取舍",
    "核心术语": "查得快、看得懂的深度学习术语表",
    "训练全流程": "一个模型从数据到参数更新的完整路径",
    "Transformer": "从入门直觉到论文原文",
}

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^\s`]*)\s*$")
INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1", re.S)
BLOCK_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SLOT_RE = re.compile(r'<span class="math-slot" data-mi="(\d+)"></span>')
LIST_RE = re.compile(r"^([-*+]|\d+[.)])\s+")

# 占位符字符: 使用 Unicode 私有区, 避开 markdown 内部占位符 (\u0002/\u0003)
PH_L, PH_R = "\ue000", "\ue001"


def normalize_lists(text: str) -> str:
    """在紧跟段落的列表块前补空行, 让 Python-Markdown 正确识别为列表"""
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    fence_mark = ""
    prev = ""
    for line in lines:
        stripped = line.strip()
        if in_fence:
            out.append(line)
            if stripped.startswith(fence_mark):
                in_fence = False
            prev = line
            continue
        m = FENCE_RE.match(line)
        if m:
            in_fence = True
            fence_mark = m.group(2)[:3]
            out.append(line)
            prev = line
            continue
        p = prev.strip()
        if (LIST_RE.match(stripped) and not line.startswith((" ", "\t")) and p
                and not LIST_RE.match(p) and not p.startswith(("|", ">"))):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
class Slugger:
    """标题文本 -> 稳定锚点 id, 同名标题自动加序号"""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def slug(self, text: str) -> str:
        s = re.sub(r"[^\w]+", "-", text.strip().lower()).strip("-")[:64].strip("-")
        s = s or "sec"
        n = self.seen.get(s, 0)
        self.seen[s] = n + 1
        return s if n == 0 else f"{s}-{n}"


def md_heading_text(raw: str) -> str:
    """还原 markdown 标题行的纯文本, 与渲染后 HTML 的文本保持一致"""
    t = re.sub(r"<[^>]+>", "", raw)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[*_`]", "", t)
    return t.strip()


def clean_index_text(s: str) -> str:
    """清理 markdown 标记, 得到用于搜索的纯文本"""
    s = re.sub(r"<img[^>]*/?>", " ", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\\[\[\]\(\)]", " ", s)
    s = re.sub(r"\$\$?", "", s)
    s = re.sub(r"[*_`~]", "", s)
    s = re.sub(r"^\s*[>#\-\+\d\.]+\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def render_code_block(lang: str, code: str) -> str:
    """Pygments 高亮 + 语言标签 + 复制按钮"""
    code = code.rstrip("\n")
    lexer = None
    if lang:
        try:
            lexer = get_lexer_by_name(lang, stripnl=False)
        except ClassNotFound:
            lexer = None
    if lexer is None:
        try:
            lexer = guess_lexer(code)
        except ClassNotFound:
            lexer = get_lexer_by_name("text")
    body = pyg_highlight(code, lexer, HtmlFormatter(nowrap=True))
    label = {
        "python": "Python", "py": "Python", "bash": "Bash", "sh": "Shell",
        "shell": "Shell", "json": "JSON", "text": "Text", "lua": "Lua",
        "cpp": "C++", "c": "C", "yaml": "YAML", "yml": "YAML", "sql": "SQL",
    }.get((lang or "").lower(), (lang or "Text").capitalize())
    esc = htmllib.escape(label, quote=True)
    return (
        f'<div class="code-block" data-lang="{esc}">'
        f'<div class="code-bar"><span class="code-lang">{esc}</span>'
        f'<button class="code-copy" type="button">复制</button></div>'
        f"<pre><code>{body}</code></pre></div>"
    )


def render_mermaid(code: str) -> str:
    return (
        '<div class="mermaid-wrap"><div class="mermaid">'
        + htmllib.escape(code, quote=False)
        + "</div></div>"
    )


# --------------------------------------------------------------------------
# 文本保护: 围栏代码块 / 行内代码 / 数学公式
# --------------------------------------------------------------------------
def protect_fences(text: str):
    lines = text.split("\n")
    out: list[str] = []
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if m:
            fence, lang = m.group(2), m.group(3)
            body: list[str] = []
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith(fence[0] * 3):
                    i += 1
                    break
                body.append(lines[i])
                i += 1
            key = f"{PH_L}B{len(blocks)}{PH_R}"
            blocks.append(dict(lang=lang, code="\n".join(body)))
            # 前后补空行, 保证代码块独占一段而不是嵌进 <p>
            out.extend(["", key, ""])
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out), blocks


def protect_inline_code(text: str):
    codes: list[str] = []

    def rep(m):
        codes.append(m.group(2))
        return f"{PH_L}I{len(codes) - 1}{PH_R}"

    return INLINE_CODE_RE.sub(rep, text), codes


MATH_RULES = [
    (re.compile(r"\$\$(.+?)\$\$", re.S), True),
    (re.compile(r"\\\[(.+?)\\\]", re.S), True),
    (re.compile(r"\\\((.+?)\\\)", re.S), False),
    (re.compile(r"(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)"), False),
]


def protect_math(text: str):
    maths: list[dict] = []

    def rep_factory(display: bool):
        def rep(m):
            maths.append(dict(tex=m.group(1).strip(), display=display))
            return f"{PH_L}M{len(maths) - 1}{PH_R}"

        return rep

    for pattern, display in MATH_RULES:
        text = pattern.sub(rep_factory(display), text)
    return text, maths


# --------------------------------------------------------------------------
# Markdown -> HTML
# --------------------------------------------------------------------------
class DocTreeprocessor(Treeprocessor):
    """标题降一级 + 生成锚点 id, 同时收集目录"""

    def __init__(self, md, prefix: str, slugger: Slugger, collector: list):
        super().__init__(md)
        self.prefix = prefix
        self.slugger = slugger
        self.collector = collector

    def run(self, root):
        for el in list(root.iter()):
            if el.tag in HEADING_TAGS:
                text = "".join(el.itertext()).strip()
                if not text:
                    continue
                lvl = int(el.tag[1])
                slug = self.slugger.slug(text)
                anchor = f"{self.prefix}--{slug}"
                el.tag = f"h{min(lvl + 1, 6)}"
                el.set("id", anchor)
                el.set("class", "doc-h")
                self.collector.append(dict(id=anchor, level=min(lvl + 1, 6), text=text))
        return root


def render_markdown(md_text: str, page_id: str, slugger: Slugger, toc: list, math_queue: list) -> str:
    protected = normalize_lists(md_text)
    protected, blocks = protect_fences(protected)
    protected, inline_codes = protect_inline_code(protected)
    protected, maths = protect_math(protected)

    base = len(math_queue)
    math_queue.extend(maths)

    md = markdown.Markdown(
        extensions=["extra", "sane_lists"],
        extension_configs={"md_in_html": {"markdown_attribute": True}},
    )
    md.treeprocessors.register(DocTreeprocessor(md, page_id, slugger, toc), "doc_anchor", 5)
    out = md.convert(protected)

    def put_block(key: str, block_html: str):
        nonlocal out
        out = re.sub(r"<p>\s*" + re.escape(key) + r"\s*</p>", lambda m: block_html, out)
        out = out.replace(key, block_html)

    for idx, blk in enumerate(blocks):
        key = f"{PH_L}B{idx}{PH_R}"
        put_block(key, render_mermaid(blk["code"]) if blk["lang"] == "mermaid"
                  else render_code_block(blk["lang"], blk["code"]))

    for idx, code in enumerate(inline_codes):
        out = out.replace(f"{PH_L}I{idx}{PH_R}", f"<code>{htmllib.escape(code, quote=False)}</code>")

    for i in range(len(maths)):
        key = f"{PH_L}M{i}{PH_R}"
        slot = f'<span class="math-slot" data-mi="{base + i}"></span>'
        out = re.sub(r"<p>\s*" + re.escape(key) + r"\s*</p>",
                     lambda m: f'<div class="math-block">{slot}</div>', out)
        out = out.replace(key, slot)

    return out


def render_math_batch(maths: list[dict]) -> list[dict]:
    if not maths:
        return []
    script = BUILD_DIR / "render_math.cjs"
    proc = subprocess.run(
        [NODE_BIN, str(script)],
        input=json.dumps(maths).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit("KaTeX 渲染失败")
    return json.loads(proc.stdout.decode("utf-8"))


# --------------------------------------------------------------------------
# 搜索索引
# --------------------------------------------------------------------------
def build_search_index(md_text: str, page_id: str, toc: list) -> list[dict]:
    """从 markdown 源生成 (锚点, 标题路径, 正文) 索引条目; toc 提供锚点序列"""
    entries: list[dict] = []
    cursor = 0
    headings: list[tuple[int, str]] = [(0, "")]
    anchors: list[str] = [page_id]
    buf: list[str] = []
    in_fence = False
    fence_mark = ""

    def flush():
        nonlocal buf
        if not buf:
            return
        text = clean_index_text(" ".join(buf))
        buf = []
        if len(text) < 2:
            return
        entries.append(dict(
            a=anchors[-1],
            h=" › ".join(t for _, t in headings[1:]),
            x=text,
        ))

    for line in md_text.split("\n"):
        stripped = line.strip()
        if in_fence:
            if stripped.startswith(fence_mark):
                in_fence = False
            elif stripped:
                buf.append(stripped)
            continue
        m = FENCE_RE.match(line)
        if m:
            flush()
            in_fence = True
            fence_mark = m.group(2)[:3]
            continue
        hm = BLOCK_HEAD_RE.match(line)
        if hm:
            flush()
            lvl = len(hm.group(1))
            text = md_heading_text(hm.group(2))
            if not text:
                continue
            if cursor < len(toc):
                anchors.append(toc[cursor]["id"])
                cursor += 1
            else:
                anchors.append(anchors[-1])
            while headings and headings[-1][0] >= lvl:
                headings.pop()
            headings.append((lvl, text))
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            if buf and buf[0].startswith("|"):
                buf.append(stripped)
                continue
            flush()
            buf.append(stripped)
            continue
        if LIST_RE.match(stripped):
            flush()
            buf.append(stripped)
            continue
        buf.append(stripped)
    flush()
    return entries


# --------------------------------------------------------------------------
# 资源
# --------------------------------------------------------------------------
def collect_images() -> dict:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    for src_dir in [ROOT_DIR / "assets", ROOT_DIR / "名词解释.assets"]:
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                shutil.copy2(f, IMG_DIR / f.name)
                mapping[f.name] = f"assets/img/{f.name}"
    return mapping


def rewrite_assets(text: str, img_map: dict) -> str:
    def fix(url: str) -> str:
        base = Path(unquote(url)).name
        return img_map.get(base, url)

    text = re.sub(r"(!\[[^\]]*\]\()([^)\s]+)(\))",
                  lambda m: m.group(1) + fix(m.group(2)) + m.group(3), text)
    text = re.sub(r'(<img[^>]*\ssrc=")([^"]+)(")',
                  lambda m: m.group(1) + fix(m.group(2)) + m.group(3), text)
    text = re.sub(r'\s*style="zoom:[^"]*"', "", text)
    return text


def slice_source(text: str, start: str | None, end: str | None) -> str:
    if start is None:
        return text
    lines = text.split("\n")
    si = None
    for i, ln in enumerate(lines):
        if ln.strip() == start:
            si = i
            break
    if si is None:
        raise SystemExit(f"找不到起始标题: {start}")
    ei = len(lines)
    if end:
        for j in range(si + 1, len(lines)):
            if lines[j].strip() == end:
                ei = j
                break
    return "\n".join(lines[si + 1: ei])


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    img_map = collect_images()
    sources = {}
    for p in PAGES:
        if p["src"] not in sources:
            sources[p["src"]] = (ROOT_DIR / p["src"]).read_text(encoding="utf-8")

    math_queue: list[dict] = []
    pages_out: dict[str, dict] = {}
    index_out: list[dict] = []
    warnings: list[str] = []

    for p in PAGES:
        raw = slice_source(sources[p["src"]], p["start"], p["end"])
        raw = rewrite_assets(raw, img_map)
        slugger = Slugger()
        toc: list[dict] = []
        html_out = render_markdown(raw, p["id"], slugger, toc, math_queue)
        entries = build_search_index(raw, p["id"], toc)
        for e in entries:
            index_out.append(dict(p=p["id"], a=e["a"], h=e["h"], x=e["x"]))
        pages_out[p["id"]] = dict(
            id=p["id"], title=p["title"], desc=p["desc"], badge=p["badge"],
            group=p["group"], toc=toc, html=html_out,
            chars=sum(len(e["x"]) for e in entries),
        )

    # ---- KaTeX 服务端回填 ----
    rendered = render_math_batch(math_queue)
    fail = 0
    def math_html(i: int) -> str:
        r = rendered[i]
        item = math_queue[i]
        cls = "math-display" if item["display"] else "math-inline"
        if r["ok"]:
            return f'<span class="{cls}">{r["html"]}</span>'
        return ('<span class="math-error" title="'
                + htmllib.escape(r["error"][:120], quote=True) + '">'
                + htmllib.escape(item["tex"][:120]) + "</span>")

    for page in pages_out.values():
        page["html"] = SLOT_RE.sub(lambda m: math_html(int(m.group(1))), page["html"])
    for i, r in enumerate(rendered):
        if not r["ok"]:
            fail += 1
            warnings.append(f"公式 #{i}: {r['error'][:60]} | {math_queue[i]['tex'][:70]}")

    # ---- 分组与元信息 ----
    order: list[str] = []
    groups: dict[str, list[str]] = {}
    for p in PAGES:
        groups.setdefault(p["group"], [])
        if p["group"] not in order:
            order.append(p["group"])
        groups[p["group"]].append(p["id"])

    payload = dict(
        meta=dict(
            title="深度学习框架学习手册",
            subtitle="框架 · 术语 · 训练流程 · Transformer 论文，一处检索",
            built=datetime.now().strftime("%Y-%m-%d"),
            stats=dict(
                pages=len(PAGES), groups=len(order),
                terms=sum(1 for t in pages_out["glossary"]["toc"] if t["level"] == 3),
                entries=len(index_out), math=len(math_queue), images=len(img_map),
            ),
        ),
        groups=[dict(name=g, desc=GROUP_DESC.get(g, ""), pages=groups[g]) for g in order],
        pages=pages_out,
        index=index_out,
    )

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    js = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    js = js.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    OUT_JS.write_text("window.SITE_DATA=" + js + ";\n", encoding="utf-8")

    # ---- 自检 ----
    leftovers = []
    for pid, page in pages_out.items():
        if PH_L in page["html"] or PH_R in page["html"]:
            leftovers.append(pid)
    print(f"✓ 页面 {len(PAGES)} 个 | 索引 {len(index_out)} 条 | 公式 {len(math_queue)} 个 | 图片 {len(img_map)} 张")
    print(f"✓ 输出 {OUT_JS.relative_to(SITE_DIR)} ({OUT_JS.stat().st_size / 1024:.0f} KB)")
    if leftovers:
        print("! 占位符未完全替换:", ", ".join(leftovers))
    if fail:
        print(f"! {fail} 个公式渲染异常:")
        for w in warnings[:15]:
            print("   -", w)


if __name__ == "__main__":
    main()
