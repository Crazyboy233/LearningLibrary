/* ==========================================================================
   深度学习框架学习手册 — 前端逻辑
   纯静态实现：hash 路由 · 全文搜索 · 主题切换 · Mermaid 懒加载
   ========================================================================== */
(function () {
  'use strict';

  var DATA = window.SITE_DATA;
  if (!DATA) { console.error('content.js 未加载'); return; }

  var $ = function (id) { return document.getElementById(id); };

  var el = {
    progress: $('progress'), nav: $('nav'), content: $('content'), foot: $('foot'),
    tocNav: $('tocNav'), tocAside: $('tocAside'), tocAll: $('tocAll'),
    tocBtn: $('tocBtn'), tocClose: $('tocClose'),
    sideResizer: $('sideResizer'), tocResizer: $('tocResizer'), sideMeta: $('sideMeta'),
    search: $('search'), searchInput: $('searchInput'), searchPop: $('searchPop'),
    searchList: $('searchList'), searchHead: $('searchHead'), searchClear: $('searchClear'),
    searchKbd: $('searchKbd'), themeBtn: $('themeBtn'), menuBtn: $('menuBtn'),
    mask: $('mask'), sidebar: $('sidebar'), toTop: $('toTop'),
    lightbox: $('lightbox'), lbBar: document.querySelector('.lb-bar'),
    lbViewport: $('lbViewport'), lbStage: $('lbStage'), lbScale: $('lbScale')
  };

  var PAGES = DATA.pages;
  var ORDER = [];
  DATA.groups.forEach(function (g) { g.pages.forEach(function (p) { ORDER.push(p); }); });

  var currentPage = null;
  var pendingQuery = null;

  /* ---------------------------------------------------------------- 工具 */
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function isMac() { return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent); }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy') ? resolve() : reject(); }
      catch (e) { reject(e); }
      finally { document.body.removeChild(ta); }
    });
  }

  /* ---------------------------------------------------------------- 主题 */
  function currentTheme() { return document.documentElement.getAttribute('data-theme') || 'light'; }

  function setTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem('dlf-theme', t); } catch (e) { /* ignore */ }
    reRenderMermaid();
  }

  el.themeBtn.addEventListener('click', function () {
    setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  });

  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onSys = function (e) {
      var saved = null;
      try { saved = localStorage.getItem('dlf-theme'); } catch (err) { /* ignore */ }
      if (!saved) setTheme(e.matches ? 'dark' : 'light');
    };
    mq.addEventListener ? mq.addEventListener('change', onSys) : mq.addListener(onSys);
  }

  /* ---------------------------------------------------------------- 路由 */
  function parseHash() {
    var h = location.hash.replace(/^#\/?/, '');
    if (!h) return { page: '', anchor: '' };
    var i = h.indexOf('/');
    if (i < 0) return { page: decodeURIComponent(h), anchor: '' };
    return { page: decodeURIComponent(h.slice(0, i)), anchor: decodeURIComponent(h.slice(i + 1)) };
  }

  function go(page, anchor, query) {
    if (query) pendingQuery = query;
    var want = '#/' + page + (anchor ? '/' + anchor : '');
    if (location.hash === want) { route(); }
    else { location.hash = want; }
  }

  function route() {
    var r = parseHash();
    var pid = PAGES[r.page] ? r.page : '';
    if (pid === currentPage) {
      scrollToAnchor(r.anchor, true);
      return;
    }
    currentPage = pid;
    render(pid, r.anchor);
  }

  /* ------------------------------------------------------------ 侧边导航 */
  function buildNav() {
    var html = '';
    DATA.groups.forEach(function (g) {
      html += '<div class="nav-group"><div class="nav-group-title">' + esc(g.name) + '</div>';
      g.pages.forEach(function (pid) {
        var p = PAGES[pid];
        html += '<a class="nav-item" href="#/' + pid + '" data-page="' + pid + '">'
          + '<span class="nav-dot"></span><span class="n">' + esc(p.title) + '</span>'
          + '<span class="nav-sub">' + (p.toc.length ? p.toc.length : '') + '</span></a>';
      });
      html += '</div>';
    });
    el.nav.innerHTML = html;
  }

  function markNav(pid) {
    el.nav.querySelectorAll('.nav-item').forEach(function (a) {
      a.classList.toggle('on', a.dataset.page === pid);
    });
  }

  /* ---------------------------------------------------------------- 首页 */
  function renderHome() {
    var s = DATA.meta.stats;
    var hot = (PAGES.glossary ? PAGES.glossary.toc : [])
      .filter(function (t) { return t.level === 3; })
      .slice(0, 18);

    var html = ''
      + '<section class="hero">'
      + '<h1>深度学习框架<br><span class="grad">学习手册</span></h1>'
      + '<p>' + esc(DATA.meta.subtitle) + '。把散落的笔记汇总成一本可检索的小册子。</p>'
      + '<div class="hero-search" id="heroSearch">'
      + '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="M20 20l-3.6-3.6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
      + '<span>搜索 ' + s.terms + ' 个术语 · ' + s.entries + ' 条笔记</span>'
      + '<kbd>' + (isMac() ? '⌘' : 'Ctrl') + ' K</kbd></div>'
      + '<div class="stats">'
      + stat(s.pages, '篇笔记') + stat(s.terms, '个术语') + stat(s.math, '个公式')
      + stat(s.images, '张配图') + stat(s.groups, '个专题')
      + '</div></section>';

    html += '<div class="sec-title">按专题浏览</div><div class="cards">';
    DATA.groups.forEach(function (g) {
      var links = g.pages.map(function (pid) {
        return '<a class="card-link" href="#/' + pid + '">' + esc(PAGES[pid].title) + '</a>';
      }).join('');
      html += '<div class="card"><div class="card-top"><span class="card-dot"></span>'
        + '<span class="card-name">' + esc(g.name) + '</span></div>'
        + '<div class="card-desc">' + esc(g.desc || '') + '</div>'
        + '<div class="card-links">' + links + '</div></div>';
    });
    html += '</div>';

    if (hot.length) {
      html += '<div class="sec-title">术语速查</div><div class="chips open"><div class="chips-body">'
        + hot.map(function (t) {
          return '<a class="chip" href="#/glossary/' + t.id + '">' + esc(t.text) + '</a>';
        }).join('')
        + '</div></div>';
    }

    el.content.innerHTML = html;
    el.content.className = 'wrap';
    el.tocNav.innerHTML = '';
    el.tocAside.hidden = true;
    el.tocAll.hidden = true;
    tocItems = [];
    currentTocId = null;
    updateTocBtn();
    el.foot.innerHTML = footText();
    markNav(null);
    document.title = DATA.meta.title;

    var hs = $('heroSearch');
    if (hs) hs.addEventListener('click', function () { openSearch(); });

    el.content.querySelectorAll('.chip[href^="#/"]').forEach(function (a) {
      a.addEventListener('click', function () { pendingQuery = null; });
    });
  }

  function stat(v, k) { return '<div class="stat"><div class="stat-v">' + v + '</div><div class="stat-k">' + k + '</div></div>'; }
  function footText() {
    var s = DATA.meta.stats;
    return s.pages + ' 篇笔记 · ' + s.entries + ' 条索引 · ' + s.math + ' 个公式';
  }

  /* ------------------------------------------------------------ 内容页面 */
  function render(pid, anchor) {
    if (!pid) { renderHome(); window.scrollTo(0, 0); return; }

    var p = PAGES[pid];
    var s = DATA.meta.stats;
    var body = p.html;

    var isEmpty = p.chars < 30;
    var html = '<article><header class="page-head">'
      + '<div class="page-badge">' + esc(p.badge || p.group) + '</div>'
      + '<h1>' + esc(p.title) + '</h1>'
      + '<p class="page-desc">' + esc(p.desc) + '</p></header>';

    // 术语索引 chips
    var terms = p.toc.filter(function (t) { return t.level === 3; });
    if (terms.length >= 12) {
      html += '<div class="chips" id="chips"><div class="chips-head">'
        + '<span class="chips-title">速查索引 · ' + terms.length + ' 项</span>'
        + '<button class="chips-toggle" type="button">展开全部</button></div>'
        + '<div class="chips-body">'
        + terms.map(function (t) {
          return '<a class="chip" href="#/' + pid + '/' + t.id + '">' + esc(t.text) + '</a>';
        }).join('')
        + '</div></div>';
    }

    html += isEmpty
      ? '<div class="empty-state"><strong>这一节还在整理中</strong>内容尚未落笔，先去其他页面看看吧。</div>'
      : '<div class="doc-body">' + body + '</div>';

    // 上下页
    var idx = ORDER.indexOf(pid);
    var prev = idx > 0 ? PAGES[ORDER[idx - 1]] : null;
    var next = idx >= 0 && idx < ORDER.length - 1 ? PAGES[ORDER[idx + 1]] : null;
    if (prev || next) {
      html += '<nav class="page-nav">'
        + (prev ? '<a class="prev" href="#/' + prev.id + '"><span class="dir">← 上一篇</span><span class="ttl">' + esc(prev.title) + '</span></a>' : '')
        + (next ? '<a class="next" href="#/' + next.id + '"><span class="dir">下一篇 →</span><span class="ttl">' + esc(next.title) + '</span></a>' : '')
        + '</nav>';
    }
    html += '</article>';

    el.content.innerHTML = html;
    el.content.className = 'wrap';
    el.foot.innerHTML = footText();
    document.title = p.title + ' · ' + DATA.meta.title;
    markNav(pid);

    afterRender(pid, anchor);
  }

  function afterRender(pid, anchor) {
    var p = PAGES[pid];

    // 表格横向滚动包裹
    el.content.querySelectorAll('table').forEach(function (t) {
      if (t.parentElement && t.parentElement.classList.contains('table-scroll')) return;
      var w = document.createElement('div');
      w.className = 'table-scroll';
      t.parentNode.insertBefore(w, t);
      w.appendChild(t);
    });

    // 外链新窗口打开
    el.content.querySelectorAll('a[href^="http"]').forEach(function (a) {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    });

    // 内部锚点（markdown 里可能出现的 #xxx）
    el.content.querySelectorAll('a[href^="#"]:not([href^="#/"])').forEach(function (a) {
      a.addEventListener('click', function (e) {
        var id = a.getAttribute('href').slice(1);
        var target = document.getElementById(id);
        if (target) { e.preventDefault(); scrollToEl(target, true); }
      });
    });

    // 术语 chips 展开
    var chips = $('chips');
    if (chips) {
      var toggle = chips.querySelector('.chips-toggle');
      if (pid === 'glossary') chips.classList.remove('open');
      toggle.addEventListener('click', function () {
        var open = chips.classList.toggle('open');
        toggle.textContent = open ? '收起' : '展开全部';
      });
      chips.querySelectorAll('.chip').forEach(function (a) {
        a.addEventListener('click', function () { pendingQuery = null; });
      });
    }

    // 图片灯箱
    el.content.querySelectorAll('.doc-body img').forEach(function (img) {
      img.addEventListener('click', function () {
        openLightbox('<img src="' + esc(img.getAttribute('src')) + '" alt="' + esc(img.alt || '') + '">');
      });
    });

    buildToc(p);
    renderMermaid(el.content);

    if (pendingQuery) {
      var q = pendingQuery;
      pendingQuery = null;
      jumpToMatch(anchor, q);
    } else {
      // 首次进入页面时瞬时定位，避免长页面平滑滚动造成错位
      scrollToAnchor(anchor, false, true);
    }
  }

  /* ------------------------------------------------------------- 目录 */
  var tocItems = [];   // 当前页目录项（含对应 DOM 节点），用于滚动定位

  function tocTree(items) {
    var root = { level: 0, children: [] };
    var stack = [root];
    items.forEach(function (t) {
      while (stack.length > 1 && stack[stack.length - 1].level >= t.level) stack.pop();
      var node = { id: t.id, text: t.text, level: t.level, children: [] };
      stack[stack.length - 1].children.push(node);
      stack.push(node);
    });
    return root;
  }

  function tocHtml(node) {
    var out = '';
    node.children.forEach(function (c) {
      var kids = c.children.length ? '<div class="toc-kids">' + tocHtml(c) + '</div>' : '';
      out += '<div class="toc-node" data-node="' + esc(c.id) + '">'
        + '<div class="toc-row">'
        + (c.children.length
          ? '<button class="toc-tgl" type="button" aria-expanded="false" aria-label="展开子目录"></button>'
          : '<span class="toc-sp"></span>')
        + '<a class="toc-link lv-' + c.level + '" href="#/' + currentPage + '/' + esc(c.id) + '"'
        + ' data-id="' + esc(c.id) + '" title="' + esc(c.text) + '">' + esc(c.text) + '</a>'
        + '</div>' + kids + '</div>';
    });
    return out;
  }

  function buildToc(p) {
    var items = p.toc.filter(function (t) { return t.level >= 2 && t.level <= 5; });
    currentTocId = null;
    el.tocAside.hidden = !items.length;
    if (!items.length) { el.tocNav.innerHTML = ''; tocItems = []; el.tocAll.hidden = true; updateTocBtn(); return; }

    tocItems = items.map(function (t) {
      return { id: t.id, text: t.text, level: t.level, node: document.getElementById(t.id) };
    }).filter(function (t) { return t.node; });

    el.tocNav.innerHTML = tocHtml(tocTree(items));
    el.tocNav.querySelectorAll('.toc-node').forEach(openIfCurrent);
    updateTocAllBtn();
    updateTocBtn();
    scheduleTocHighlight();
  }

  function hasKids(n) {
    var kids = n.querySelector('.toc-kids');
    return !!(kids && kids.querySelector('.toc-node'));
  }

  function setOpen(node, open) {
    if (!hasKids(node)) return;
    node.classList.toggle('open', open);
    var tgl = node.querySelector('.toc-tgl');
    if (tgl) tgl.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  function openIfCurrent(node) {
    var link = node.querySelector('.toc-link');
    if (link && link.dataset.id === currentTocId) setOpen(node, true);
  }

  function openAncestors(link) {
    var node = link.closest('.toc-node');
    var parent = node && node.parentElement ? node.parentElement.closest('.toc-node') : null;
    while (parent) {
      if (!parent.classList.contains('open')) {
        setOpen(parent, true);
        updateTocAllBtn();
      }
      parent = parent.parentElement ? parent.parentElement.closest('.toc-node') : null;
    }
  }

  function updateTocAllBtn() {
    var nodes = [].filter.call(el.tocNav.querySelectorAll('.toc-node'), hasKids);
    if (!nodes.length) { el.tocAll.hidden = true; return; }
    el.tocAll.hidden = false;
    var allOpen = nodes.every(function (n) { return n.classList.contains('open'); });
    el.tocAll.textContent = allOpen ? '折叠' : '展开';
  }

  el.tocAll.addEventListener('click', function () {
    var nodes = [].filter.call(el.tocNav.querySelectorAll('.toc-node'), hasKids);
    var allOpen = nodes.length && nodes.every(function (n) { return n.classList.contains('open'); });
    nodes.forEach(function (n) { setOpen(n, !allOpen); });
    el.tocAll.textContent = allOpen ? '展开' : '折叠';
  });

  el.tocNav.addEventListener('click', function (e) {
    var tgl = e.target.closest('.toc-tgl');
    if (tgl) {
      e.preventDefault();
      var node = tgl.closest('.toc-node');
      setOpen(node, !node.classList.contains('open'));
      updateTocAllBtn();
      return;
    }
    if (e.target.closest('.toc-link')) { pendingQuery = null; closeToc(); }
  });

  var tocTick = false;
  var currentTocId = null;

  function scheduleTocHighlight() {
    if (tocTick) return;
    tocTick = true;
    requestAnimationFrame(function () { tocTick = false; highlightToc(); });
  }

  function highlightToc() {
    if (!tocItems.length) return;
    var baseLine = window.scrollY + 100;
    var currentId = tocItems[0].id;
    for (var i = 0; i < tocItems.length; i++) {
      if (tocItems[i].node.getBoundingClientRect().top + window.scrollY <= baseLine) currentId = tocItems[i].id;
      else break;
    }
    if (currentId === currentTocId && el.tocNav.querySelector('.toc-link.on')) return;
    currentTocId = currentId;

    var links = el.tocNav.querySelectorAll('.toc-link');
    var target = null;
    for (var j = 0; j < links.length; j++) {
      var on = links[j].dataset.id === currentId;
      links[j].classList.toggle('on', on);
      if (on) target = links[j];
    }
    if (!target) return;

    // 滚动到当前章节时，自动展开它所在的父级分支
    openAncestors(target);

    var box = el.tocAside;
    if (!box.offsetParent) return;
    var r = target.getBoundingClientRect();
    var br = box.getBoundingClientRect();
    if (r.top < br.top + 52) box.scrollTop -= (br.top + 52 - r.top);
    else if (r.bottom > br.bottom - 24) box.scrollTop += (r.bottom - br.bottom + 24);
  }

  /* ------------------------------------------------------------- 滚动 */
  function scrollToEl(target, flash, instant) {
    if (!target) return;
    var y = target.getBoundingClientRect().top + window.scrollY - 76;
    window.scrollTo({ top: Math.max(0, y), behavior: instant ? 'auto' : 'smooth' });
    if (flash) {
      target.classList.remove('flash');
      void target.offsetWidth;
      target.classList.add('flash');
      setTimeout(function () { target.classList.remove('flash'); }, 1600);
    }
  }

  function scrollToAnchor(anchor, flash, instant) {
    if (!anchor) { window.scrollTo({ top: 0, behavior: instant ? 'auto' : 'smooth' }); return; }
    var t = document.getElementById(anchor);
    if (t) scrollToEl(t, flash, instant);
  }

  /* ------------------------------------------- 搜索命中定位 + 正文高亮 */
  var TEXT_SKIP = /^(CODE|PRE|SCRIPT|STYLE|MARK)$/;
  function walkTextNodes(root, cb) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (node) {
        var p = node.parentNode;
        while (p && p !== root) {
          if (TEXT_SKIP.test(p.nodeName) || (p.classList && (p.classList.contains('katex') || p.classList.contains('mermaid')))) {
            return NodeFilter.FILTER_REJECT;
          }
          p = p.parentNode;
        }
        return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    var n;
    while ((n = walker.nextNode())) cb(n);
  }

  function clearMarks(root) {
    root.querySelectorAll('mark').forEach(function (m) {
      var parent = m.parentNode;
      if (!parent) return;
      parent.replaceChild(document.createTextNode(m.textContent), m);
      parent.normalize();
    });
  }

  function highlightTerms(root, terms, limit) {
    limit = limit || 300;
    clearMarks(root);
    var nodes = [];
    walkTextNodes(root, function (n) { nodes.push(n); });
    var count = 0;

    for (var k = 0; k < nodes.length && count < limit; k++) {
      var node = nodes[k];
      var text = node.nodeValue;
      var low = text.toLowerCase();
      var has = false;
      for (var i = 0; i < terms.length; i++) { if (low.indexOf(terms[i]) >= 0) { has = true; break; } }
      if (!has) continue;

      var frag = document.createDocumentFragment();
      var idx = 0;
      while (idx < text.length && count < limit) {
        var best = -1, bestLen = 0;
        for (var j = 0; j < terms.length; j++) {
          var pos = low.indexOf(terms[j], idx);
          if (pos >= 0 && (best < 0 || pos < best)) { best = pos; bestLen = terms[j].length; }
        }
        if (best < 0) break;
        if (best > idx) frag.appendChild(document.createTextNode(text.slice(idx, best)));
        var mk = document.createElement('mark');
        mk.textContent = text.slice(best, best + bestLen);
        frag.appendChild(mk);
        idx = best + bestLen;
        count++;
      }
      if (idx < text.length) frag.appendChild(document.createTextNode(text.slice(idx)));
      node.parentNode.replaceChild(frag, node);
    }
    return count;
  }

  function isAtOrAfter(node, base) {
    var pos = base.compareDocumentPosition(node);
    return !!(pos & (Node.DOCUMENT_POSITION_FOLLOWING | Node.DOCUMENT_POSITION_CONTAINED_BY));
  }

  function jumpToMatch(anchor, query) {
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    var scope = el.content.querySelector('.doc-body') || el.content;
    if (!terms.length) { scrollToAnchor(anchor, true); return; }

    highlightTerms(scope, terms);

    var startEl = anchor ? document.getElementById(anchor) : null;
    var marks = scope.querySelectorAll('mark');
    var target = null;
    for (var i = 0; i < marks.length; i++) {
      if (!startEl || isAtOrAfter(marks[i], startEl)) { target = marks[i]; break; }
    }
    if (!target) target = marks[0] || startEl;
    if (target) scrollToEl(target, true);
    else scrollToAnchor(anchor, true);
  }

  /* ------------------------------------------------------------- 搜索 */

  /**
   * 索引按「页面 + 锚点」合并后再用于检索。
   *
   * 构建期是逐段落切片的，同一章节会拆成多条片段（如「过拟合和欠拟合」一节
   * 被拆成 26 条），但它们的跳转锚点是同一个，直接罗列就会出现好几条
   * 「点了跳到同一处」的结果。这里把同锚点的片段按原文顺序拼回一条，
   * 让「一条结果 = 一个落点」，段落顺序即章节内的行文顺序。
   */
  var INDEX = (function () {
    var groups = {};
    var merged = [];
    DATA.index.forEach(function (e) {
      var key = e.p + '\u0000' + e.a;
      var g = groups[key];
      if (g) { g.x += ' ' + e.x; return; }
      groups[key] = g = { p: e.p, a: e.a, h: e.h, x: e.x };
      merged.push(g);
    });
    return merged.map(function (e) {
      var p = PAGES[e.p];
      return {
        p: e.p, a: e.a, h: e.h, x: e.x,
        lx: e.x.toLowerCase(),
        lh: (e.h || '').toLowerCase(),
        lt: ((p ? p.title : '') + ' ' + (p ? p.badge : '')).toLowerCase(),
        pt: p ? p.title : e.p
      };
    });
  })();

  var sel = -1;
  var results = [];

  /**
   * 在纯文本里高亮所有命中词。
   * 先算出所有匹配区间并合并，再统一转义拼装，避免二次扫描时改坏已插入的标签。
   */
  function hl(text, terms) {
    if (!text) return '';
    if (!terms.length) return esc(text);
    var low = text.toLowerCase();
    var spans = [];
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i];
      if (!t) continue;
      var from = 0, at;
      while ((at = low.indexOf(t, from)) >= 0) {
        spans.push([at, at + t.length]);
        from = at + t.length;
      }
    }
    if (!spans.length) return esc(text);
    spans.sort(function (a, b) { return a[0] - b[0] || b[1] - a[1]; });
    var merged = [spans[0]];
    for (var j = 1; j < spans.length; j++) {
      var last = merged[merged.length - 1];
      if (spans[j][0] <= last[1]) last[1] = Math.max(last[1], spans[j][1]);
      else merged.push(spans[j]);
    }
    var out = '', cur = 0;
    merged.forEach(function (sp) {
      out += esc(text.slice(cur, sp[0]));
      out += '<mark>' + esc(text.slice(sp[0], sp[1])) + '</mark>';
      cur = sp[1];
    });
    return out + esc(text.slice(cur));
  }

  function snippet(text, terms) {
    var low = text.toLowerCase();
    var pos = -1;
    for (var i = 0; i < terms.length; i++) {
      var p = low.indexOf(terms[i]);
      if (p >= 0 && (pos < 0 || p < pos)) pos = p;
    }
    var has = pos >= 0;
    if (!has) pos = 0;
    var start = has ? Math.max(0, pos - 36) : 0;
    var end = Math.min(text.length, pos + 130);
    var out = (start > 0 ? '…' : '') + text.slice(start, end) + (end < text.length ? '…' : '');
    return hl(out, terms);
  }

  function scoreEntry(e, terms, mode) {
    var score = 0, hits = 0;
    for (var j = 0; j < terms.length; j++) {
      var t = terms[j];
      var inTitle = e.lt.indexOf(t) >= 0;
      var inHead = e.lh.indexOf(t) >= 0;
      var at = e.lx.indexOf(t);
      if (!inTitle && !inHead && at < 0) continue;
      hits++;
      if (inTitle) score += 70;
      if (inHead) score += 45;
      if (at >= 0) score += 12 + Math.max(0, 10 - Math.floor(at / 60));
    }
    if (mode === 'and') return hits === terms.length ? score : -1;
    return hits ? score + hits * 8 : -1;
  }

  var lastRelaxed = false;

  function collect(terms, mode, limit) {
    var out = [];
    for (var i = 0; i < INDEX.length; i++) {
      var e = INDEX[i];
      var s = scoreEntry(e, terms, mode);
      if (s < 0) continue;
      // 「整段文字就是这个关键词」的短词条（多为术语），以及标题路径恰好等于关键词时加权。
      // 合并后片段已拼成章节，故同时用标题判断，保留原有的词条优先效果。
      if (terms.length === 1 && (e.lh === terms[0] || e.x.length <= terms[0].length + 2)) s += 25;
      out.push({ e: e, s: s });
    }
    out.sort(function (a, b) { return b.s - a.s || a.e.x.length - b.e.x.length; });
    return out.slice(0, 60).map(function (r) { return r.e; });
  }

  function runSearch(q) {
    var terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) { lastRelaxed = false; return []; }
    var out = collect(terms, 'and');
    lastRelaxed = false;
    if (!out.length && terms.length > 1) {
      out = collect(terms, 'or');
      lastRelaxed = true;
    }
    return out;
  }

  function renderResults(q) {
    var terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    results = runSearch(q);
    sel = results.length ? 0 : -1;

    el.searchHead.innerHTML = results.length
      ? '<span>找到 <strong>' + results.length + '</strong> 条结果'
        + (lastRelaxed ? '<span style="opacity:.75"> · 已放宽为任一关键词</span>' : '') + '</span>'
        + '<span>' + esc(q) + '</span>'
      : '<span>没有匹配结果</span>';

    if (!results.length) {
      el.searchList.innerHTML = '<div class="search-empty">换个关键词试试，例如 <b>Flash Attention</b>、<b>ZeRO</b>、<b>KV Cache</b></div>';
      return;
    }
    el.searchList.innerHTML = results.map(function (e, i) {
      var body = snippet(e.x, terms);
      var onlyTitle = body.indexOf('<mark>') < 0;
      return '<div class="search-item' + (i === sel ? ' sel' : '') + '" data-i="' + i + '" role="option">'
        + '<div class="search-crumb"><span class="pill">' + hl(e.pt, terms) + '</span>'
        + (onlyTitle && e.h ? '<span class="hit-tag">标题匹配</span>' : '')
        + (e.h ? '<span class="path" title="' + esc(e.h) + '">' + hl(e.h, terms) + '</span>' : '')
        + '</div>'
        + '<div class="search-snippet' + (onlyTitle ? ' dim' : '') + '">' + body + '</div></div>';
    }).join('');
  }

  function openSearch(preset) {
    if (preset) el.searchInput.value = preset;
    el.searchPop.hidden = false;
    if (el.searchInput.value.trim()) renderResults(el.searchInput.value);
    el.searchInput.focus();
    el.searchInput.select();
  }
  function closeSearch() { el.searchPop.hidden = true; sel = -1; }

  function selectResult(i) {
    var items = el.searchList.querySelectorAll('.search-item');
    if (!items.length) return;
    sel = (i + items.length) % items.length;
    for (var k = 0; k < items.length; k++) items[k].classList.toggle('sel', k === sel);
    items[sel].scrollIntoView({ block: 'nearest' });
  }

  function commit(index) {
    var e = results[index];
    if (!e) return;
    var q = el.searchInput.value.trim();
    closeSearch();
    el.searchInput.blur();
    if (e.p === currentPage) {
      jumpToMatch(e.a, q);
      history.replaceState(null, '', '#/' + e.p + '/' + e.a);
    } else {
      go(e.p, e.a, q);
    }
    closeNav();
  }

  var searchTimer = null;
  el.searchInput.addEventListener('input', function () {
    var v = el.searchInput.value;
    el.searchClear.hidden = !v;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function () {
      if (!v.trim()) { el.searchPop.hidden = true; return; }
      el.searchPop.hidden = false;
      renderResults(v);
    }, 110);
  });

  el.searchInput.addEventListener('focus', function () {
    if (el.searchInput.value.trim()) openSearch();
  });

  el.searchInput.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); if (el.searchPop.hidden) openSearch(); else selectResult(sel + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); selectResult(sel - 1); }
    else if (e.key === 'Enter') { e.preventDefault(); if (results.length) commit(sel >= 0 ? sel : 0); }
    else if (e.key === 'Escape') { closeSearch(); el.searchInput.blur(); }
  });

  el.searchClear.addEventListener('click', function () {
    el.searchInput.value = '';
    el.searchClear.hidden = true;
    closeSearch();
    el.searchInput.focus();
  });

  el.searchList.addEventListener('click', function (e) {
    var item = e.target.closest('.search-item');
    if (item) commit(Number(item.dataset.i));
  });
  el.searchList.addEventListener('mousemove', function (e) {
    var item = e.target.closest('.search-item');
    if (item && Number(item.dataset.i) !== sel) selectResult(Number(item.dataset.i));
  });

  document.addEventListener('click', function (e) {
    if (!el.search.contains(e.target)) closeSearch();
  });

  document.addEventListener('keydown', function (e) {
    var mod = isMac() ? e.metaKey : e.ctrlKey;
    if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); el.searchPop.hidden ? openSearch() : closeSearch(); return; }
    if (e.key === '/' && document.activeElement !== el.searchInput && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
      e.preventDefault(); openSearch();
    }
  });

  /* --------------------------------------------------------- Mermaid */
  var mermaidLoading = null;
  function loadMermaid() {
    if (window.mermaid) return Promise.resolve(window.mermaid);
    if (mermaidLoading) return mermaidLoading;
    mermaidLoading = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = 'assets/vendor/mermaid.min.js';
      s.onload = function () { resolve(window.mermaid); };
      s.onerror = reject;
      document.head.appendChild(s);
    });
    return mermaidLoading;
  }

  function mermaidConfig(theme) {
    var dark = theme === 'dark';
    return {
      startOnLoad: false,
      securityLevel: 'strict',
      theme: dark ? 'dark' : 'default',
      fontFamily: '-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif',
      fontSize: 15,
      flowchart: { curve: 'basis', useMaxWidth: false, htmlLabels: false, padding: 12, nodeSpacing: 32, rankSpacing: 42 },
      themeVariables: dark ? {
        background: 'transparent',
        primaryColor: '#1e2530', primaryTextColor: '#e6e9ef', primaryBorderColor: '#4b5563',
        lineColor: '#7c8798', secondaryColor: '#232a35', tertiaryColor: '#1a2029',
        mainBkg: '#1a2029', nodeBorder: '#4b5563', clusterBkg: '#141922',
        clusterBorder: '#333c4a', titleColor: '#e6e9ef', edgeLabelBackground: '#171a21'
      } : {
        background: 'transparent',
        primaryColor: '#eef1ff', primaryTextColor: '#1f2430', primaryBorderColor: '#c7cdf5',
        lineColor: '#94a0b4', secondaryColor: '#f4f6fb', tertiaryColor: '#fbfcfe',
        mainBkg: '#f7f9ff', nodeBorder: '#c7cdf5', clusterBkg: '#fafbff',
        clusterBorder: '#dfe4f0', titleColor: '#1f2430', edgeLabelBackground: '#ffffff'
      }
    };
  }

  function renderMermaid(scope) {
    var nodes = scope.querySelectorAll('.mermaid:not([data-rendered])');
    if (!nodes.length) return;
    var jobs = [];
    nodes.forEach(function (n) {
      if (!n.dataset.src) n.dataset.src = n.textContent;
      jobs.push(n);
      n.innerHTML = '<div class="mermaid-loading">正在绘制流程图…</div>';
    });
    loadMermaid().then(function (m) {
      m.initialize(mermaidConfig(currentTheme()));
      jobs.forEach(function (n) {
        var src = n.dataset.src;
        try {
          m.render('mmd-' + Math.random().toString(36).slice(2), src).then(function (res) {
            n.innerHTML = res.svg;
            n.setAttribute('data-rendered', '1');
            var svg = n.querySelector('svg');
            var box = n.parentNode;
            if (svg && svg.getBBox) {
              // mermaid 会画一块撑满 viewBox 的背景矩形，先隐掉再测真实边界
              var bgRect = svg.querySelector('rect.background, .root > rect.background');
              if (bgRect) bgRect.style.display = 'none';
              try {
                var bb = svg.getBBox();
                var pad = 10;
                var w = Math.ceil(bb.width + pad * 2), h = Math.ceil(bb.height + pad * 2);
                if (bb.width > 0 && bb.height > 0) {
                  svg.setAttribute('viewBox', (bb.x - pad) + ' ' + (bb.y - pad) + ' ' + w + ' ' + h);
                  svg.setAttribute('width', w);
                  svg.setAttribute('height', h);
                }
              } catch (e) { /* ignore */ }
              if (bgRect) bgRect.style.display = '';
            }
            if (box && svg) {
              enableWrapPan(box);
              box.onclick = function () {
                openLightbox('<div class="lightbox-svg">' + svg.outerHTML + '</div>');
              };
              requestAnimationFrame(function () {
                if (box.scrollWidth > box.clientWidth + 8 && !box.querySelector('.mermaid-hint')) {
                  var hint = document.createElement('div');
                  hint.className = 'mermaid-hint';
                  hint.textContent = '← 左右拖动查看完整流程图 · 点击放大';
                  box.insertBefore(hint, box.firstChild);
                }
              });
            }
          }).catch(function (err) { fallback(n, src, err); });
        } catch (err) { fallback(n, src, err); }
      });
    }).catch(function () {
      jobs.forEach(function (n) { fallback(n, n.dataset.src, null); });
    });
  }

  var lb = { x: 0, y: 0, scale: 1, min: 0.05, max: 12, dragging: false, sx: 0, sy: 0 };

  function lbApply() {
    el.lbStage.style.transform = 'translate(' + lb.x + 'px,' + lb.y + 'px) scale(' + lb.scale + ')';
    el.lbScale.textContent = Math.round(lb.scale * 100) + '%';
  }

  function lbSize() {
    var n = el.lbStage.firstElementChild;
    return n ? { w: n.offsetWidth, h: n.offsetHeight } : null;
  }

  // 缩放到刚好放得下整个内容
  function lbFit() {
    var size = lbSize();
    if (!size || !size.w) return;
    var vw = el.lbViewport.clientWidth, vh = el.lbViewport.clientHeight;
    var pad = 28;
    lb.scale = Math.max(lb.min, Math.min((vw - pad * 2) / size.w, (vh - pad * 2) / size.h, 1));
    lb.x = (vw - size.w * lb.scale) / 2;
    lb.y = (vh - size.h * lb.scale) / 2;
    lbApply();
  }

  // 原始像素大小并居中
  function lbActual() {
    var size = lbSize() || { w: 0, h: 0 };
    lb.scale = 1;
    lb.x = (el.lbViewport.clientWidth - size.w) / 2;
    lb.y = (el.lbViewport.clientHeight - size.h) / 2;
    lbApply();
  }

  function lbZoomAt(cx, cy, factor) {
    var next = Math.min(lb.max, Math.max(lb.min, lb.scale * factor));
    var k = next / lb.scale;
    lb.x = cx - (cx - lb.x) * k;
    lb.y = cy - (cy - lb.y) * k;
    lb.scale = next;
    lbApply();
  }

  function lbZoomCenter(factor) {
    lbZoomAt(el.lbViewport.clientWidth / 2, el.lbViewport.clientHeight / 2, factor);
  }

  // 页面内的宽图：按住拖动即可平移，拖动后不触发放大
  function enableWrapPan(wrap) {
    if (!wrap || wrap.dataset.panReady) return;
    wrap.dataset.panReady = '1';
    var down = false, moved = false, sx = 0, sy = 0, sl = 0, st = 0;
    wrap.addEventListener('pointerdown', function (e) {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      down = true; moved = false;
      sx = e.clientX; sy = e.clientY;
      sl = wrap.scrollLeft; st = wrap.scrollTop;
    });
    wrap.addEventListener('pointermove', function (e) {
      if (!down) return;
      var dx = e.clientX - sx, dy = e.clientY - sy;
      if (!moved && Math.abs(dx) + Math.abs(dy) > 5) {
        moved = true;
        wrap.classList.add('panning');
        try { wrap.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      }
      if (moved) {
        wrap.scrollLeft = sl - dx;
        wrap.scrollTop = st - dy;
        e.preventDefault();
      }
    });
    function up(e) {
      if (!down) return;
      down = false;
      if (!moved) return;
      wrap.classList.remove('panning');
      try { wrap.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      wrap.dataset.panned = '1';
      setTimeout(function () { delete wrap.dataset.panned; }, 0);
    }
    wrap.addEventListener('pointerup', up);
    wrap.addEventListener('pointercancel', up);
    wrap.addEventListener('click', function (e) {
      if (wrap.dataset.panned) { e.preventDefault(); e.stopPropagation(); }
    }, true);
  }

  function openLightbox(inner) {
    el.lbStage.innerHTML = inner;
    el.lightbox.hidden = false;
    lb.x = 0; lb.y = 0; lb.scale = 1;
    var node = el.lbStage.firstElementChild;
    var ready = function () { requestAnimationFrame(lbFit); };
    if (node && node.tagName === 'IMG' && !node.complete) {
      node.addEventListener('load', ready, { once: true });
      node.addEventListener('error', ready, { once: true });
    } else {
      ready();
    }
  }

  function closeLightbox() {
    el.lightbox.hidden = true;
    el.lbStage.innerHTML = '';
  }

  // 滚轮 / 触控板捏合缩放（以指针位置为中心）
  el.lbViewport.addEventListener('wheel', function (e) {
    e.preventDefault();
    var r = el.lbViewport.getBoundingClientRect();
    var k = e.ctrlKey ? 0.012 : 0.0022;
    lbZoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * k));
  }, { passive: false });

  // 拖动平移
  el.lbViewport.addEventListener('pointerdown', function (e) {
    if (e.target.closest('.lb-bar')) return;
    lb.dragging = true;
    lb.sx = e.clientX - lb.x;
    lb.sy = e.clientY - lb.y;
    el.lbViewport.classList.add('dragging');
    try { el.lbViewport.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    e.preventDefault();
  });
  el.lbViewport.addEventListener('pointermove', function (e) {
    if (!lb.dragging) return;
    lb.x = e.clientX - lb.sx;
    lb.y = e.clientY - lb.sy;
    lbApply();
  });
  function lbStopDrag(e) {
    if (!lb.dragging) return;
    lb.dragging = false;
    el.lbViewport.classList.remove('dragging');
    try { el.lbViewport.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  }
  el.lbViewport.addEventListener('pointerup', lbStopDrag);
  el.lbViewport.addEventListener('pointercancel', lbStopDrag);

  // 双击在「适应屏幕 / 1:1」之间切换
  el.lbViewport.addEventListener('dblclick', function (e) {
    e.preventDefault();
    if (lb.scale < 0.999) lbActual(); else lbFit();
  });

  el.lbBar.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-act]');
    if (!btn) return;
    var act = btn.dataset.act;
    if (act === 'in') lbZoomCenter(1.3);
    else if (act === 'out') lbZoomCenter(1 / 1.3);
    else if (act === 'fit') lbFit();
    else if (act === 'one') lbActual();
    else if (act === 'close') closeLightbox();
  });

  function fallback(n, src, err) {
    n.innerHTML = '<div style="font-size:12.5px;color:var(--text-mute);margin-bottom:8px">流程图渲染失败，以下为源码：</div>'
      + '<pre class="mermaid-fallback">' + esc(src) + '</pre>';
    if (err) console.warn('mermaid:', err);
  }

  function reRenderMermaid() {
    var nodes = el.content.querySelectorAll('.mermaid[data-rendered]');
    if (!nodes.length) return;
    nodes.forEach(function (n) { n.removeAttribute('data-rendered'); });
    renderMermaid(el.content);
  }

  /* ------------------------------------------------------- 抽屉 / 其他 */
  function updateMask() {
    var open = document.body.classList.contains('nav-open') || document.body.classList.contains('toc-open');
    el.mask.hidden = !open;
  }
  function openNav() {
    document.body.classList.remove('toc-open');
    document.body.classList.add('nav-open');
    updateMask();
  }
  function closeNav() {
    document.body.classList.remove('nav-open');
    updateMask();
  }
  function openToc() {
    document.body.classList.remove('nav-open');
    document.body.classList.add('toc-open');
    updateMask();
  }
  function closeToc() {
    document.body.classList.remove('toc-open');
    updateMask();
  }
  function toggleToc() {
    document.body.classList.contains('toc-open') ? closeToc() : openToc();
  }

  // 目录按钮：只要有本页目录就显示
  //   常驻（toc-inline）—— 折叠 / 展开右侧目录，与左上角按钮折叠左侧栏对称
  //   抽屉（非 toc-inline）—— 打开 / 关闭右侧目录抽屉
  function updateTocBtn() {
    var hasToc = !el.tocAside.hidden && tocItems.length > 0;
    el.tocBtn.hidden = !hasToc;
    if (!hasToc) closeToc();
    syncTocBtn();
  }

  function syncTocBtn() {
    var root = document.documentElement;
    var inline = root.classList.contains('toc-inline');
    var collapsed = root.classList.contains('toc-collapsed');
    var label = !inline ? '本页目录' : (collapsed ? '展开右侧目录' : '收起右侧目录');
    el.tocBtn.title = label;
    el.tocBtn.setAttribute('aria-label', label);
    el.tocBtn.setAttribute('aria-expanded', inline && !collapsed ? 'true' : 'false');
  }

  function setTocCollapsed(on) {
    var root = document.documentElement;
    root.classList.toggle('toc-collapsed', on);
    try { localStorage.setItem('dlf-toc-collapsed', on ? '1' : '0'); } catch (e) { /* ignore */ }
  }

  // 宽屏：折叠 / 展开左侧栏；窄屏：抽屉开关
  function isWide() { return window.innerWidth >= 1024; }

  function syncMenuBtn() {
    var collapsed = document.documentElement.classList.contains('nav-collapsed');
    el.menuBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  }

  el.menuBtn.addEventListener('click', function () {
    if (isWide()) {
      var collapsed = document.documentElement.classList.toggle('nav-collapsed');
      try { localStorage.setItem('dlf-nav-collapsed', collapsed ? '1' : '0'); } catch (e) { /* ignore */ }
      if (window.__fitLayout) window.__fitLayout();
      updateTocBtn();
      syncMenuBtn();
    } else {
      document.body.classList.contains('nav-open') ? closeNav() : openNav();
    }
  });
  syncMenuBtn();
  syncTocBtn();
  el.tocBtn.addEventListener('click', function () {
    var root = document.documentElement;
    if (root.classList.contains('toc-inline')) {
      var collapsed = !root.classList.contains('toc-collapsed');
      setTocCollapsed(collapsed);
      if (window.__fitLayout) window.__fitLayout();
      // 展开后若空间已放不下常驻目录，直接以抽屉形式呈现
      if (!collapsed && !root.classList.contains('toc-inline')) openToc();
      syncTocBtn();
    } else {
      toggleToc();
    }
  });
  el.tocClose.addEventListener('click', closeToc);
  el.mask.addEventListener('click', function () { closeNav(); closeToc(); });

  var resizeTimer = null;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (window.__fitLayout) window.__fitLayout();
      updateTocBtn();
      syncMenuBtn();
    }, 120);
  });

  /* ------------------------------------------------ 侧栏 / 目录宽度拖拽 */
  function clampNum(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  function initResizer(handle, cssVar, storeKey, compute) {
    if (!handle) return;
    var dragging = false;

    handle.addEventListener('pointerdown', function (e) {
      if (e.button !== 0 && e.pointerType === 'mouse') return;
      dragging = true;
      handle.classList.add('dragging');
      document.body.classList.add('resizing');
      e.preventDefault();
    });

    window.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      document.documentElement.style.setProperty(cssVar, compute(e.clientX) + 'px');
      if (window.__fitLayout) window.__fitLayout();
      e.preventDefault();
    }, { passive: false });

    function stop() {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove('dragging');
      document.body.classList.remove('resizing');
      var raw = parseFloat(document.documentElement.style.getPropertyValue(cssVar));
      if (!isNaN(raw)) { try { localStorage.setItem(storeKey, String(Math.round(raw))); } catch (e) {} }
    }
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);

    // 双击复位
    handle.addEventListener('dblclick', function () {
      document.documentElement.style.removeProperty(cssVar);
      try { localStorage.removeItem(storeKey); } catch (e) {}
    });
  }

  function layoutWidth() { return document.documentElement.clientWidth || window.innerWidth; }

  initResizer(el.sideResizer, '--side-w', 'dlf-side-w', function (x) {
    return Math.round(clampNum(x, 200, Math.min(460, layoutWidth() * 0.4)));
  });
  initResizer(el.tocResizer, '--toc-w', 'dlf-toc-w', function (x) {
    return Math.round(clampNum(layoutWidth() - x, 170, Math.min(420, layoutWidth() * 0.4)));
  });

  window.addEventListener('scroll', function () {
    var h = document.documentElement.scrollHeight - window.innerHeight;
    el.progress.style.width = (h > 0 ? Math.min(100, (window.scrollY / h) * 100) : 0) + '%';
    el.toTop.hidden = window.scrollY < 700;
    scheduleTocHighlight();
  }, { passive: true });

  el.toTop.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: 'smooth' }); });

  el.lightbox.addEventListener('click', function (e) {
    if (e.target === el.lightbox) closeLightbox();
  });
  document.addEventListener('keydown', function (e) {
    if (!el.lightbox.hidden) {
      var step = 70;
      if (e.key === 'ArrowLeft') { lb.x += step; lbApply(); e.preventDefault(); return; }
      if (e.key === 'ArrowRight') { lb.x -= step; lbApply(); e.preventDefault(); return; }
      if (e.key === 'ArrowUp') { lb.y += step; lbApply(); e.preventDefault(); return; }
      if (e.key === 'ArrowDown') { lb.y -= step; lbApply(); e.preventDefault(); return; }
      if (e.key === '+' || e.key === '=') { lbZoomCenter(1.3); e.preventDefault(); return; }
      if (e.key === '-' || e.key === '_') { lbZoomCenter(1 / 1.3); e.preventDefault(); return; }
      if (e.key === '0') { lbFit(); e.preventDefault(); return; }
      if (e.key === '1') { lbActual(); e.preventDefault(); return; }
    }
    if (e.key === 'Escape') { closeLightbox(); closeSearch(); closeNav(); closeToc(); }
  });

  el.nav.addEventListener('click', function (e) {
    if (e.target.closest('.nav-item')) { pendingQuery = null; closeNav(); }
  });

  // 代码复制
  el.content.addEventListener('click', function (e) {
    var btn = e.target.closest('.code-copy');
    if (!btn) return;
    var code = btn.closest('.code-block').querySelector('pre code');
    copyText(code ? code.textContent : '').then(function () {
      btn.textContent = '已复制';
      btn.classList.add('done');
      setTimeout(function () { btn.textContent = '复制'; btn.classList.remove('done'); }, 1600);
    }).catch(function () { btn.textContent = '复制失败'; setTimeout(function () { btn.textContent = '复制'; }, 1600); });
  });

  window.addEventListener('hashchange', route);

  /* ------------------------------------------------------------- 启动 */
  buildNav();
  if (window.__fitLayout) window.__fitLayout();
  el.sideMeta.textContent = DATA.meta.stats.pages + ' 篇 · ' + DATA.meta.stats.entries + ' 条索引';
  if (el.searchKbd) el.searchKbd.textContent = (isMac() ? '⌘' : 'Ctrl') + ' K';
  route();
  try {
    var initQ = new URLSearchParams(location.search).get('q');
    if (initQ) setTimeout(function () { openSearch(initQ); }, 40);
  } catch (e) { /* ignore */ }
})();
