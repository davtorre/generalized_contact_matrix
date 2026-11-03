"""Interactive HTML plots for ContactMatrix objects.

The output is one self-contained HTML file (plotly.js from cdnjs, data embedded as JSON),
in the same style as the reports in ../contact_matrices: no Python plotting dependency.

    from gcm import heatmap
    heatmap(cm, "polymod.html", title="POLYMOD", sort_part=["country", "gender", "age"])
    heatmap(cm, "polymod_by_country.html", title="POLYMOD", block_level="country")
"""
import html
import json
import math
import webbrowser
from pathlib import Path

PLOTLY_URL = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"


def _fmt(v):
    """Feature value as text; None is the placeholder for a feature added after the fact."""
    return "any" if v is None else str(v)


def _sort_key(label, positions):
    # None last; type name before value so ints and strings are never compared directly
    return tuple((label[p] is None, type(label[p]).__name__, label[p] if label[p] is not None else 0)
                 for p in positions)


def _order(labels, domain, sort_by):
    """Indices of labels, sorted by the features in sort_by (then by the remaining ones)."""
    if sort_by is None:
        return list(range(len(labels)))
    unknown = [f for f in sort_by if f not in domain]
    if unknown:
        raise ValueError(f"cannot sort by {unknown}: features are {domain}")
    positions = [domain.index(f) for f in sort_by] + [i for i, f in enumerate(domain) if f not in sort_by]
    return sorted(range(len(labels)), key=lambda i: _sort_key(labels[i], positions))


def _features(domain):
    """Domain names of a Domain (or of a plain list of names)."""
    return list(getattr(domain, "domain_names", domain))


def _split(labels, domain, feature):
    """Group positions 0..len(labels)-1 by the value of `feature`, keeping their order.

    Returns (blocks, short_labels): blocks is [(value, [positions])] sorted by value, or a single
    (None, all positions) block when the feature is not in this domain; short_labels are the
    labels without the block feature (it is shown in the panel title instead).
    """
    if feature not in domain:
        return [(None, list(range(len(labels))))], [[_fmt(v) for v in l] for l in labels]
    p = domain.index(feature)
    groups = {}
    for k, l in enumerate(labels):
        groups.setdefault(l[p], []).append(k)
    values = sorted(groups, key=lambda v: _sort_key((v,), [0]))
    short = [[_fmt(v) for q, v in enumerate(l) if q != p] for l in labels]
    return [(v, groups[v]) for v in values], short


def heatmap(cm, path=None, *, title="Contact matrix", value="per_capita", log=False,
            sort_part=None, sort_cont=None, block_level=None, open_browser=False):
    """Write an interactive heatmap of a ContactMatrix to an HTML file.

    Rows are participant groups a, columns are contact groups b. Hovering a cell shows both
    groups, R_ab, N_part[a], N_cnt[b] and M_ab = R_ab / N_part[a]. Controls on the page switch
    between R_ab and M_ab and between a linear and a log colour scale.

    cm           a ContactMatrix (anything with to_numpy(), N_part, N_cont,
                 domain_part, domain_cont)
    path         output .html file; default "<title>.html" in the current directory
    value        initial colour value: "per_capita" (M_ab) or "contacts" (R_ab)
    log          start with a log colour scale (zero cells are drawn in the lowest colour)
    sort_part    feature names to sort rows by, e.g. ["country", "age"]; default: as stored
    sort_cont    same for the columns
    block_level  a feature name: draw one panel per value of that feature (small multiples)
                 instead of one big matrix. A participant-only feature (e.g. "country") splits
                 the rows; a contact-only one splits the columns; a feature on both sides
                 (e.g. "gender") gives a grid of blocks F->F, F->M, M->F, M->M. Panels share
                 one colour scale by default; the page has a toggle for per-panel scales.
    open_browser open the file once written

    Returns the path of the written file.
    """
    if value not in ("per_capita", "contacts"):
        raise ValueError(f"value must be 'per_capita' or 'contacts', not {value!r}")
    feat_part, feat_cont = _features(cm.domain_part), _features(cm.domain_cont)
    if block_level is not None and block_level not in feat_part and block_level not in feat_cont:
        raise ValueError(f"block_level {block_level!r} is not a feature: participant features are "
                         f"{feat_part}, contact features are {feat_cont}")

    R, labels_part, labels_cont = cm.to_numpy()
    if R.size == 0:
        raise ValueError("the contact matrix is empty")
    rows = _order(labels_part, feat_part, sort_part)
    cols = _order(labels_cont, feat_cont, sort_cont)

    def number(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    # panels: one per (participant block, contact block); a single panel without block_level
    part_blocks, part_short = _split([labels_part[i] for i in rows], feat_part, block_level)
    cont_blocks, cont_short = _split([labels_cont[j] for j in cols], feat_cont, block_level)
    split_part, split_cont = block_level in feat_part, block_level in feat_cont
    panels = []
    for pv, prow in part_blocks:
        for cv, pcol in cont_blocks:
            if split_part and split_cont:
                name = f"{block_level} {_fmt(pv)} → {_fmt(cv)}"
            elif split_part:
                name = f"{block_level} {_fmt(pv)}"
            elif split_cont:
                name = f"contacts {block_level} {_fmt(cv)}"
            else:
                name = ""
            panels.append({"title": name, "rows": prow, "cols": pcol})

    data = {
        "R": [[int(R[i, j]) for j in cols] for i in rows],
        # groups of the domain without data have no N: shown as n/a
        "Np": [number(cm.N_part.get(labels_part[i])) for i in rows],
        "Nc": [number(cm.N_cont.get(labels_cont[j])) for j in cols],
        "part": [[_fmt(v) for v in labels_part[i]] for i in rows],
        "cont": [[_fmt(v) for v in labels_cont[j]] for j in cols],
        "partShort": part_short,
        "contShort": cont_short,
        "partFeat": [str(f) for f in feat_part],
        "contFeat": [str(f) for f in feat_cont],
        "panels": panels,
        # blocks on both sides: lay the panels out as a matrix (participant blocks x contact blocks)
        "gridCols": len(cont_blocks) if split_part and split_cont else 0,
        "value": value,
        "log": bool(log),
    }
    # JSON inside <script>: escape "</" so a label can never close the tag
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

    page = (_TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__SUBTITLE__", html.escape(
                f"{len(rows)} participant groups ({', '.join(map(str, feat_part))}) × "
                f"{len(cols)} contact groups ({', '.join(map(str, feat_cont))}) · "
                f"{int(R.sum()):,} contacts"
                + (f" · {len(panels)} blocks by {block_level}" if block_level is not None else "")))
            .replace("__PLOTLY__", PLOTLY_URL)
            .replace("__DATA__", payload))

    path = Path(path) if path else Path(f"{title.strip().replace(' ', '_') or 'contact_matrix'}.html")
    path.write_text(page, encoding="utf-8")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--rule:#e4e3df;--card:#ffffff;--accent:#2a78d6;
  --seq:#cde2fb,#9ec5f4,#6da7ec,#3987e5,#256abf,#184f95,#0d366b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--rule:#33332f;--card:#222220;--accent:#3987e5;
  --seq:#0d366b,#184f95,#256abf,#3987e5,#6da7ec,#9ec5f4,#cde2fb}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--rule:#33332f;--card:#222220;--accent:#3987e5;
  --seq:#0d366b,#184f95,#256abf,#3987e5,#6da7ec,#9ec5f4,#cde2fb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1200px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:24px;line-height:1.2;margin:0 0 4px}
.note{color:var(--ink2);font-size:13.5px;margin:0}
.controls{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;margin:16px 0 8px;font-size:13.5px;color:var(--ink2)}
.seg{display:inline-flex;border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.seg button{font:inherit;border:0;background:var(--card);color:var(--ink2);padding:4px 12px;cursor:pointer}
.seg button+button{border-left:1px solid var(--rule)}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
label{cursor:pointer;display:inline-flex;gap:6px;align-items:center}
.scroll{overflow-x:auto}
#hm{display:grid;gap:28px 20px;grid-template-columns:repeat(auto-fill,minmax(min(440px,100%),1fr))}
#hm.single{grid-template-columns:1fr}
figure{margin:0;min-width:0}
figcaption{font-weight:600;font-size:14px;color:var(--ink)}
[hidden]{display:none!important}
details{margin-top:16px} summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:14px}
.tbl{max-height:480px;overflow:auto;border:1px solid var(--rule);border-radius:6px;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:3px 10px;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);color:var(--ink2);font-weight:600}
td.n{text-align:right}
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="note">__SUBTITLE__. Rows: participant group <i>a</i>; columns: contact group <i>b</i>. Hover a cell for its values.</p>
<div class="controls">
  <span>Colour by</span>
  <span class="seg" role="group" aria-label="Colour by">
    <button type="button" data-value="per_capita">M<sub>ab</sub> per participant</button>
    <button type="button" data-value="contacts">R<sub>ab</sub> total contacts</button>
  </span>
  <label><input type="checkbox" id="log"> log colour scale</label>
  <label id="shared-wrap" hidden><input type="checkbox" id="shared"> same colour scale for all blocks</label>
</div>
<div class="scroll"><div id="hm"></div></div>
<details id="tv"><summary>Table view</summary><div class="tbl"><table><thead><tr id="th"></tr></thead><tbody id="tb"></tbody></table></div></details>
</main>
<script id="cm-data" type="application/json">__DATA__</script>
<script src="__PLOTLY__"></script>
<script>
(function(){
  const D = JSON.parse(document.getElementById('cm-data').textContent);
  const el = document.getElementById('hm');
  if (!window.Plotly) { el.textContent = 'Could not load plotly.js (offline?). The table view below still works.'; }
  const nA = D.R.length, nB = D.cont.length;
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const short = l => l.join(' · ');
  const long = (feat, l) => feat.map((f, k) => esc(f) + ' ' + esc(l[k])).join(' · ');
  const M = D.R.map((row, i) => row.map(r => D.Np[i] ? r / D.Np[i] : null));
  const fmt = (x, d) => x == null ? 'n/a' : x.toLocaleString(undefined, {maximumFractionDigits: d});

  // duplicate axis labels would be merged by plotly, so the axes use indices and ticktext
  const custom = D.R.map((row, i) => row.map((r, j) => [
    long(D.partFeat, D.part[i]), long(D.contFeat, D.cont[j]),
    fmt(r, 0), fmt(M[i][j], 3), fmt(D.Np[i], 0), fmt(D.Nc[j], 0)]));
  // axis labels leave out the block feature, which is in the panel title
  const yText = D.partShort.map(l => esc(short(l))), xText = D.contShort.map(l => esc(short(l)));

  // one figure per panel; with blocks on both sides the grid mirrors the block matrix
  const single = D.panels.length === 1;
  el.classList.toggle('single', single);
  if (D.gridCols) el.style.gridTemplateColumns = `repeat(${D.gridCols}, minmax(320px, 1fr))`;
  const divs = D.panels.map(p => {
    const fig = document.createElement('figure');
    if (p.title) { const cap = document.createElement('figcaption'); cap.textContent = p.title; fig.appendChild(cap); }
    const div = document.createElement('div'); fig.appendChild(div); el.appendChild(fig);
    return div;
  });

  let state = {value: D.value, log: D.log, shared: true};

  function colors() {
    const cs = getComputedStyle(document.documentElement), v = n => cs.getPropertyValue(n).trim();
    return {ink: v('--ink'), ink2: v('--ink2'), rule: v('--rule'), card: v('--card'), bg: v('--bg'),
            // one-hue blue ramp, near-zero end towards the surface (light in light mode, dark in dark mode)
            scale: v('--seq').split(',').map((h, k, a) => [k / (a.length - 1), h.trim()])};
  }

  // largest value and smallest positive value of a matrix (rows of numbers / nulls)
  function range(mat) {
    let hi = 0, pos = Infinity;
    mat.forEach(r => r.forEach(x => { if (x != null) { hi = Math.max(hi, x); if (x > 0) pos = Math.min(pos, x); } }));
    return {hi, pos};
  }

  // colour values and colourbar for one panel, given the range to map onto the colour scale
  function zAndBar(sub, rg) {
    const title = state.value === 'contacts' ? 'R<sub>ab</sub>' : 'M<sub>ab</sub>';
    if (!state.log || !isFinite(rg.pos))
      return {z: sub, zmin: 0, zmax: rg.hi || 1, bar: {title: {text: title}}};
    const l0 = Math.log10(rg.pos), l1 = Math.max(Math.log10(rg.hi), l0 + 1e-9);
    const z = sub.map(r => r.map(x => x == null ? null : x > 0 ? Math.log10(x) : l0));
    const tv = [], tt = [];
    for (let p = Math.ceil(l0); p <= Math.floor(l1); p++) { tv.push(p); tt.push(fmt(10 ** p, 3)); }
    if (tv.length < 2) { tv.splice(0, tv.length, l0, l1); tt.splice(0, tt.length, fmt(rg.pos, 3), fmt(rg.hi, 3)); }
    return {z, zmin: l0, zmax: l1, bar: {title: {text: title + ' (log)'}, tickvals: tv, ticktext: tt}};
  }

  function draw() {
    if (!window.Plotly) return;
    const c = colors(), raw = state.value === 'contacts' ? D.R : M;
    const subs = D.panels.map(p => p.rows.map(i => p.cols.map(j => raw[i][j])));
    const all = range(subs.flat());
    D.panels.forEach((p, k) => drawPanel(divs[k], p, subs[k], state.shared ? all : range(subs[k]), c));
  }

  function drawPanel(div, p, sub, rg, c) {
    const {z, zmin, zmax, bar} = zAndBar(sub, rg);
    const nR = p.rows.length, nC = p.cols.length;
    const yT = p.rows.map(i => yText[i]), xT = p.cols.map(j => xText[j]);
    const cell = Math.max(10, Math.min(32, (single ? 900 : 520) / nR));
    const left = 12 + 6.5 * Math.max(...yT.map(s => s.length));
    const bottom = 20 + 6.5 * Math.max(...xT.map(s => s.length));
    const trace = {
      type: 'heatmap', z, zmin, zmax, zauto: false,
      customdata: p.rows.map(i => p.cols.map(j => custom[i][j])),
      colorscale: c.scale, xgap: 1, ygap: 1, hoverongaps: false,
      hovertemplate:
        '<b>%{customdata[3]}</b> contacts per participant (M<sub>ab</sub>)<br>' +
        '<b>%{customdata[2]}</b> total contacts (R<sub>ab</sub>)<br><br>' +
        'participant <i>a</i>: %{customdata[0]}<br>  N<sub>part</sub> = %{customdata[4]}<br>' +
        'contact <i>b</i>: %{customdata[1]}<br>  N<sub>cnt</sub> = %{customdata[5]}<extra></extra>',
      colorbar: Object.assign({thickness: 12, outlinewidth: 0, lenmode: 'pixels', len: Math.min(single ? 320 : 240, cell * nR),
                               y: 1, yanchor: 'top', tickfont: {color: c.ink2, size: 11}}, bar)
    };
    trace.colorbar.title.font = {color: c.ink2, size: 12};
    // axis titles only on a single matrix; panels are explained by their caption
    const axis = (text, n, title) => ({
      tickmode: 'array', tickvals: [...Array(n).keys()], ticktext: text, tickfont: {color: c.ink2, size: 10},
      title: single ? {text: title, font: {color: c.ink2, size: 12}} : {text: ''}, showgrid: false, zeroline: false,
      ticks: '', fixedrange: false});
    const layout = {
      height: Math.round(cell * nR + bottom + (single ? 60 : 20)),
      margin: {l: left + (single ? 24 : 4), r: 10, t: 10, b: bottom},
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: c.bg,
      xaxis: Object.assign(axis(xT, nC, 'contact group b'), {side: 'bottom', tickangle: -90}),
      yaxis: Object.assign(axis(yT, nR, 'participant group a'), {autorange: 'reversed'}),
      hoverlabel: {bgcolor: c.card, bordercolor: c.rule, font: {color: c.ink, size: 12}, align: 'left'},
      font: {family: '-apple-system,Segoe UI,Helvetica,Arial,sans-serif'}
    };
    Plotly.react(div, [trace], layout, {displaylogo: false, responsive: true});
  }

  // controls
  const btns = document.querySelectorAll('.seg button'), logBox = document.getElementById('log');
  const sharedBox = document.getElementById('shared');
  document.getElementById('shared-wrap').hidden = single;
  const sync = () => { btns.forEach(b => b.setAttribute('aria-pressed', b.dataset.value === state.value));
                       logBox.checked = state.log; sharedBox.checked = state.shared; };
  btns.forEach(b => b.addEventListener('click', () => { state.value = b.dataset.value; sync(); draw(); }));
  logBox.addEventListener('change', () => { state.log = logBox.checked; draw(); });
  sharedBox.addEventListener('change', () => { state.shared = sharedBox.checked; draw(); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  sync(); draw();

  // table view, built on first open (labels via textContent)
  document.getElementById('tv').addEventListener('toggle', function once(e) {
    if (!e.target.open) return;
    e.target.removeEventListener('toggle', once);
    const th = document.getElementById('th'), tb = document.getElementById('tb');
    [...D.partFeat.map(f => 'part ' + f), ...D.contFeat.map(f => 'cnt ' + f), 'R_ab', 'N_part', 'N_cnt', 'M_ab']
      .forEach(h => { const x = document.createElement('th'); x.textContent = h; th.appendChild(x); });
    const frag = document.createDocumentFragment();
    for (let i = 0; i < nA; i++) for (let j = 0; j < nB; j++) {
      const tr = document.createElement('tr');
      const cellsTxt = [...D.part[i], ...D.cont[j]], nums = [fmt(D.R[i][j], 0), fmt(D.Np[i], 0), fmt(D.Nc[j], 0), fmt(M[i][j], 3)];
      cellsTxt.forEach(t => { const td = document.createElement('td'); td.textContent = t; tr.appendChild(td); });
      nums.forEach(t => { const td = document.createElement('td'); td.className = 'n'; td.textContent = t; tr.appendChild(td); });
      frag.appendChild(tr);
    }
    tb.appendChild(frag);
  });
})();
</script>
</body>
</html>
"""
