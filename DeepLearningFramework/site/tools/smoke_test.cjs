/**
 * 无头逻辑自检：用 jsdom 加载站点，验证路由 / 渲染 / 搜索等核心逻辑不报错
 * 用法: NODE_PATH=<workspace>/node_modules node site/build/smoke_test.cjs
 */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const SITE = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');

const errors = [];
const dom = new JSDOM(html, {
  url: 'http://127.0.0.1:8934/index.html',
  runScripts: 'dangerously',
  resources: undefined,
  pretendToBeVisual: true,
  beforeParse(win) {
    win.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
    win.requestAnimationFrame = (cb) => setTimeout(cb, 0);
    win.scrollTo = () => {};
    win.console.error = (...a) => errors.push('console.error: ' + a.join(' '));
    win.console.warn = (...a) => errors.push('console.warn: ' + a.join(' '));
    win.addEventListener('error', (e) => errors.push('window.error: ' + e.message));
  },
});

const win = dom.window;

// 手动注入脚本（避免资源加载的复杂度）
['assets/js/content.js', 'assets/js/app.js'].forEach((f) => {
  const code = fs.readFileSync(path.join(SITE, f), 'utf8');
  const s = win.document.createElement('script');
  s.textContent = code;
  win.document.head.appendChild(s);
});

const doc = win.document;
const out = [];
function check(name, cond, extra) {
  out.push((cond ? '  ✓ ' : '  ✗ ') + name + (extra ? '  → ' + extra : ''));
  if (!cond) process.exitCode = 1;
}

// --- 首页 ---
check('侧边导航已渲染', doc.querySelectorAll('#nav .nav-item').length === 12,
  doc.querySelectorAll('#nav .nav-item').length + ' 项');
check('首页 hero 已渲染', !!doc.querySelector('.hero'));
check('首页统计卡片', doc.querySelectorAll('.stats .stat').length === 5);
check('分组卡片', doc.querySelectorAll('.cards .card').length === 4);
check('导航不含已删除的微调分组', !/微调/.test(doc.getElementById('nav').textContent));
check('首页术语 chips', doc.querySelectorAll('.chips .chip').length > 0);
check('首页不含「本地静态站点」标签', !doc.querySelector('.hero-tag') && !/本地静态站点/.test(doc.body.textContent));
check('页脚不含构建日期', !/构建于|最近构建/.test(doc.getElementById('foot').textContent),
  doc.getElementById('foot').textContent.trim());

// --- 面板拖拽 ---
check('左右面板均带拖拽手柄',
  !!doc.getElementById('sideResizer') && !!doc.getElementById('tocResizer'));

const sideRes = doc.getElementById('sideResizer');
const pdown = new win.Event('pointerdown', { bubbles: true });
pdown.button = 0; pdown.pointerType = 'mouse';
sideRes.dispatchEvent(pdown);
const pmove = new win.Event('pointermove');
pmove.clientX = 320;
win.dispatchEvent(pmove);
check('拖拽可改变左侧栏宽度',
  doc.documentElement.style.getPropertyValue('--side-w') === '320px',
  doc.documentElement.style.getPropertyValue('--side-w'));
win.dispatchEvent(new win.Event('pointerup'));
check('拖拽结束后写入本地存储',
  win.localStorage.getItem('dlf-side-w') === '320',
  String(win.localStorage.getItem('dlf-side-w')));
sideRes.dispatchEvent(new win.MouseEvent('dblclick', { bubbles: true }));
check('双击可复位宽度', doc.documentElement.style.getPropertyValue('--side-w') === '');

// --- 左侧栏折叠（宽屏）---
const menuBtn = doc.getElementById('menuBtn');
check('侧栏默认展开', !doc.documentElement.classList.contains('nav-collapsed'));
menuBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
check('点击左上角按钮可折叠侧栏', doc.documentElement.classList.contains('nav-collapsed'));
check('折叠状态写入本地存储', win.localStorage.getItem('dlf-nav-collapsed') === '1');
check('按钮 aria-expanded 同步', menuBtn.getAttribute('aria-expanded') === 'false');
menuBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
check('再次点击可恢复侧栏', !doc.documentElement.classList.contains('nav-collapsed'));

// --- 切换到名词解释 ---
win.location.hash = '#/glossary';
win.dispatchEvent(new win.Event('hashchange'));
// 渲染是同步的，此处断言的是"构建完成时"的默认状态（滚动高亮在下一帧才跑）
check('目录默认全部折叠', doc.querySelectorAll('#tocNav .toc-node.open').length === 0);

setTimeout(() => {
  check('glossary 页面标题', (doc.querySelector('.page-head h1') || {}).textContent === '名词解释');
  check('glossary 目录渲染', doc.querySelectorAll('#tocNav .toc-link').length > 50,
    doc.querySelectorAll('#tocNav .toc-link').length + ' 项');
  check('目录为树形结构', doc.querySelectorAll('#tocNav .toc-node').length > 50,
    doc.querySelectorAll('#tocNav .toc-node').length + ' 个节点');
  check('有子级的节点带折叠按钮', doc.querySelectorAll('#tocNav .toc-tgl').length > 0,
    doc.querySelectorAll('#tocNav .toc-tgl').length + ' 个');
  const tglNode = doc.querySelector('#tocNav .toc-tgl').closest('.toc-node');
  doc.querySelector('#tocNav .toc-tgl').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('点击箭头可展开子级', tglNode.classList.contains('open'));
  doc.querySelector('#tocNav .toc-tgl').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('再次点击可折叠', !tglNode.classList.contains('open'));
  const allBtn = doc.getElementById('tocAll');
  allBtn.dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('“展开”按钮可一键展开', doc.querySelectorAll('#tocNav .toc-node.open').length > 5,
    doc.querySelectorAll('#tocNav .toc-node.open').length + ' 个展开');
  check('按钮文案随状态切换', allBtn.textContent === '折叠', allBtn.textContent);
  check('窄屏出现目录按钮', doc.getElementById('tocBtn').hidden === false);
  doc.getElementById('tocBtn').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('点击目录按钮打开右侧抽屉', doc.body.classList.contains('toc-open'));
  check('抽屉打开时显示遮罩', doc.getElementById('mask').hidden === false);
  doc.getElementById('tocClose').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
  check('可关闭目录抽屉', !doc.body.classList.contains('toc-open'));
  check('glossary 术语 chips', doc.querySelectorAll('#chips .chip').length > 50,
    doc.querySelectorAll('#chips .chip').length + ' 项');
  check('表格已包裹滚动容器', doc.querySelectorAll('.table-scroll table').length === 9,
    doc.querySelectorAll('.table-scroll table').length + ' 个');
  check('正文含 KaTeX 渲染结果', doc.querySelectorAll('.katex').length > 50,
    doc.querySelectorAll('.katex').length + ' 个');
  check('代码块均带复制按钮',
    doc.querySelectorAll('.code-block').length === doc.querySelectorAll('.code-block .code-copy').length
    && doc.querySelectorAll('.code-block').length > 20,
    doc.querySelectorAll('.code-block').length + ' 个');
  check('导航高亮当前页', !!doc.querySelector('.nav-item.on[data-page="glossary"]'));

  // --- 论文页 ---
  win.location.hash = '#/paper-transformer';
  win.dispatchEvent(new win.Event('hashchange'));

  setTimeout(() => {
    check('论文页首个标题不是重复标题',
      !/h2[^>]*>Attention Is All You Need</.test(doc.querySelector('.doc-body').innerHTML));
    check('论文页图片已重写路径',
      Array.from(doc.querySelectorAll('.doc-body img')).every((i) => i.getAttribute('src').startsWith('assets/img/')));

    // --- 图片 / 图表查看器：缩放 + 平移---
    doc.querySelector('.doc-body img').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
    check('点击图片可打开查看器', doc.getElementById('lightbox').hidden === false);
    const lbScale = () => parseFloat(doc.getElementById('lbScale').textContent);
    const lbTf = () => doc.getElementById('lbStage').style.transform;
    const s0 = lbScale();
    check('打开后自动适应屏幕并显示比例', s0 > 0 && s0 <= 100, s0 + '%');
    doc.querySelector('.lb-bar [data-act="in"]').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
    const s1 = lbScale();
    check('点击 + 可放大', s1 > s0, s0 + '% → ' + s1 + '%');
    doc.querySelector('.lb-bar [data-act="one"]').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
    check('1:1 按钮切到原始大小', doc.getElementById('lbScale').textContent === '100%');
    doc.querySelector('.lb-bar [data-act="fit"]').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
    check('适应屏幕按钮可复位', lbScale() <= 100);

    const vp = doc.getElementById('lbViewport');
    const tf0 = lbTf();
    const dEv = new win.Event('pointerdown', { bubbles: true });
    dEv.button = 0; dEv.pointerType = 'mouse'; dEv.clientX = 100; dEv.clientY = 100;
    vp.dispatchEvent(dEv);
    const mEv = new win.Event('pointermove', { bubbles: true });
    mEv.clientX = 190; mEv.clientY = 150;
    vp.dispatchEvent(mEv);
    check('拖动可平移画面', lbTf() !== tf0 && /translate\(90px,50px\)/.test(lbTf()), lbTf());
    vp.dispatchEvent(new win.Event('pointerup', { bubbles: true }));

    const wEv = new win.Event('wheel', { bubbles: true, cancelable: true });
    wEv.deltaY = -240; wEv.clientX = 200; wEv.clientY = 200;
    const beforeWheel = lbScale();
    vp.dispatchEvent(wEv);
    check('滚轮可缩放', lbScale() > beforeWheel, beforeWheel + '% → ' + lbScale() + '%');

    doc.querySelector('.lb-bar [data-act="close"]').dispatchEvent(new win.MouseEvent('click', { bubbles: true }));
    check('可关闭查看器', doc.getElementById('lightbox').hidden === true);

    // --- 流程图页（Mermaid 容器 + 上下页导航）---
    win.location.hash = '#/training-flow';
    win.dispatchEvent(new win.Event('hashchange'));

    setTimeout(() => {
      check('流程图容器已生成', doc.querySelectorAll('.mermaid').length === 2,
        doc.querySelectorAll('.mermaid').length + ' 个');
      check('上下页导航', doc.querySelectorAll('.page-nav a').length === 2);
      check('页面正文不含蒸馏内容', !/蒸馏|SFT 数据/.test(doc.getElementById('content').textContent));

      // --- 搜索 ---
      const input = doc.getElementById('searchInput');
      input.value = 'Flash Attention';
      input.dispatchEvent(new win.Event('input'));

      setTimeout(() => {
        const items = doc.querySelectorAll('#searchList .search-item');
        check('搜索结果非空', items.length > 0, items.length + ' 条');
        check('搜索结果含高亮', doc.querySelectorAll('#searchList mark').length > 0);
        check('搜索面板可显示', doc.getElementById('searchPop').hidden === false);

        // 多关键词
        input.value = 'KV Cache 显存';
        input.dispatchEvent(new win.Event('input'));
        setTimeout(() => {
          const items2 = doc.querySelectorAll('#searchList .search-item');
          check('多关键词搜索', items2.length > 0, items2.length + ' 条');

          // 关键词只出现在标题里（RQ）时，也要有高亮
          input.value = 'RQ';
          input.dispatchEvent(new win.Event('input'));
          setTimeout(() => {
            const crumbMarks = doc.querySelectorAll('#searchList .search-crumb mark');
            const allMarks = doc.querySelectorAll('#searchList mark');
            check('标题命中的结果也有高亮', allMarks.length > 0, allMarks.length + ' 处');
            check('高亮出现在面包屑 / 页面名上', crumbMarks.length > 0, crumbMarks.length + ' 处');
            check('仅标题命中时打「标题匹配」标记',
              doc.querySelectorAll('#searchList .hit-tag').length > 0);

            // 空结果
            input.value = 'zzzzz-not-exists';
            input.dispatchEvent(new win.Event('input'));
            setTimeout(() => {
              check('无结果提示', !!doc.querySelector('.search-empty'));

              console.log(out.join('\n'));
              if (errors.length) {
                console.log('\n控制台告警/错误:');
                errors.slice(0, 12).forEach((e) => console.log('   ! ' + e.slice(0, 180)));
                process.exitCode = 1;
              } else {
                console.log('\n无控制台错误。');
              }
              dom.window.close();
            }, 200);
          }, 200);
        }, 200);
      }, 250);
    }, 60);
  }, 60);
}, 80);
