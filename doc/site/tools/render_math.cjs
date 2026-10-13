/**
 * 构建期 KaTeX 服务端渲染
 * 从 stdin 读取 JSON: [{ tex, display }]，向 stdout 写回 [{ ok, html, error }]
 * 用法: node render_math.cjs < in.json > out.json
 */
const path = require('path');
const katex = require(path.join(__dirname, 'vendor', 'katex.min.js'));

let chunks = [];
process.stdin.on('data', (d) => chunks.push(d));
process.stdin.on('end', () => {
  const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  const out = input.map((item) => {
    try {
      const html = katex.renderToString(item.tex, {
        displayMode: !!item.display,
        throwOnError: false,
        strict: false,
        output: 'html',
        macros: { '\\RR': '\\mathbb{R}' },
      });
      // katex 内部对不支持的命令会输出 error 节点，这里做一次探测
      const failed = html.includes('katex-error');
      return { ok: !failed, html, error: failed ? 'unsupported command' : '' };
    } catch (e) {
      return { ok: false, html: '', error: String((e && e.message) || e) };
    }
  });
  process.stdout.write(JSON.stringify(out));
});
