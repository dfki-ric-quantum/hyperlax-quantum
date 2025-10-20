"""Export analysis results to CSV and HTML."""

import logging
import time
from pathlib import Path

import pandas as pd

from .metrics import SummaryMetrics

logger = logging.getLogger(__name__)


def _prettify_column_name(column: str) -> str:
    """Replace underscores with spaces for human-friendly exports."""
    return str(column).replace("_", " ")


def export_summaries_to_csv(
    summaries: list[SummaryMetrics],
    output_path: Path,
    hyperparam_mapping: dict | None = None,
):
    """Export summary metrics to CSV, with NaN-valued runs at the end."""
    if not summaries:
        return

    df = pd.DataFrame([s.to_dict() for s in summaries])

    if hyperparam_mapping:
        hp_series = {}
        for hp_id, params in hyperparam_mapping.items():
            if params is None:
                continue
            for key, value in params.items():
                col = str(key)
                hp_series.setdefault(col, {})[hp_id] = value

        for col, mapping in hp_series.items():
            if col in df.columns:
                continue
            df[col] = df["hp_id"].map(mapping)

    # Separate valid from NaN runs to sort them differently.
    nan_df = df[df['peak_return'].isna()]
    valid_df = df[df['peak_return'].notna()]

    # Sort valid runs by performance, and NaN runs by ID.
    valid_df = valid_df.sort_values('peak_return', ascending=False)
    nan_df = nan_df.sort_values('hp_id', ascending=True)

    # Combine them, with valid runs first.
    final_df = pd.concat([valid_df, nan_df], ignore_index=True)

    export_df = final_df.rename(columns=_prettify_column_name)
    export_df.to_csv(output_path, index=False, float_format='%.4g')


def generate_html_index(output_dir: Path) -> None:
    """Generate an interactive HTML dashboard with global +/- height controls and a selectable column count."""
    generated_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    index_path = (output_dir / "index.html").resolve()

    tables_dir = output_dir / "tables"
    if tables_dir.exists():
        import shutil

        shutil.rmtree(tables_dir)

    summary_htmls: list[str] = []
    slice_htmls: list[str] = []
    optuna_htmls: list[str] = []
    table_iframes: list[tuple[str, str]] = []  # (iframe src, original csv rel)

    for html_path in sorted(output_dir.rglob("*.html")):
        if html_path.resolve() == index_path:
            continue
        rel = html_path.relative_to(output_dir).as_posix()
        if rel.startswith("tables/") or rel.startswith("data/"):
            continue
        if rel.startswith("optuna/"):
            optuna_htmls.append(rel)
        elif "slice_plot" in html_path.stem:
            slice_htmls.append(rel)
        else:
            summary_htmls.append(rel)

    data_dir = output_dir / "data"
    if data_dir.exists():
        tables_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in sorted(data_dir.rglob("*.csv")):
            rel_csv = csv_path.relative_to(output_dir).as_posix()
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.warning("Failed to read CSV '%s': %s", csv_path, exc)
                continue

            safe_name = rel_csv.replace("/", "__")
            if safe_name.lower().endswith(".csv"):
                safe_name = safe_name[:-4]
            table_rel = f"tables/{safe_name}.html"
            table_path = output_dir / table_rel

            html_table = df.to_html(index=False, classes="csv-table", border=0, justify="center")
            table_doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{rel_csv}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif; margin: 1.5rem; background: #f6f8fa; }}
    h1 {{ font-size: 1.25rem; margin-bottom: 1rem; }}
    table.csv-table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
    table.csv-table thead {{ background: #eff2f6; }}
    table.csv-table th, table.csv-table td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
    table.csv-table th {{ cursor: pointer; user-select: none; position: relative; padding-right: 1.75rem; }}
    table.csv-table th .sort-indicator {{ position: absolute; right: 8px; font-size: 0.75rem; opacity: 0.6; }}
    table.csv-table tr:nth-child(even) {{ background: #fafbfc; }}
    caption {{ caption-side: top; text-align: left; font-weight: 600; margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
  <h1>{rel_csv}</h1>
  <div class=\"table-wrapper\">{html_table}</div>
  <script>
    (function() {{
      const table = document.querySelector('table.csv-table');
      if (!table) return;

      const tbody = table.tBodies[0];
      const headers = Array.from(table.tHead ? table.tHead.rows[0].cells : []);
      let currentSortColumn = -1;
      let currentSortDirection = 0; // 0 = none, 1 = asc, -1 = desc

      headers.forEach((th, index) => {{
        const span = document.createElement('span');
        span.className = 'sort-indicator';
        span.textContent = '↕';
        th.appendChild(span);

        th.addEventListener('click', () => {{
          // Reset all indicators first
          headers.forEach((h, i) => {{
            const indicator = h.querySelector('.sort-indicator');
            if (indicator) {{
              indicator.textContent = '↕';
            }}
          }});

          // Determine new sort direction
          if (currentSortColumn === index) {{
            // Same column clicked - cycle through states
            currentSortDirection = currentSortDirection === 1 ? -1 : (currentSortDirection === -1 ? 0 : 1);
          }} else {{
            // New column clicked - start with ascending
            currentSortColumn = index;
            currentSortDirection = 1;
          }}

          // Update indicator for current column
          const indicator = th.querySelector('.sort-indicator');
          if (indicator) {{
            if (currentSortDirection === 1) {{
              indicator.textContent = '↑';
            }} else if (currentSortDirection === -1) {{
              indicator.textContent = '↓';
            }} else {{
              indicator.textContent = '↕';
              currentSortColumn = -1;
            }}
          }}

          // Only sort if we have a direction
          if (currentSortDirection !== 0) {{
            const rows = Array.from(tbody.rows);
            const type = detectType(rows, index);

            rows.sort((a, b) => {{
              const result = compareCells(a.cells[index], b.cells[index], type);
              return result * currentSortDirection;
            }});

            // Re-append sorted rows
            const fragment = document.createDocumentFragment();
            rows.forEach(row => fragment.appendChild(row));
            tbody.appendChild(fragment);
          }}
        }});
      }});

      function detectType(rows, idx) {{
        let hasNumbers = false;
        let hasDates = false;

        // Sample first few non-empty cells to determine type
        for (let i = 0; i < Math.min(rows.length, 10); i++) {{
          const cell = rows[i].cells[idx];
          if (!cell) continue;

          const text = cell.textContent.trim();
          if (!text || text.toLowerCase() === 'nan' || text === '') continue;

          // Check for date first (more specific)
          if (isValidDate(text)) {{
            hasDates = true;
          }} else if (isNumeric(text)) {{
            hasNumbers = true;
          }}
        }}

        if (hasDates) return 'date';
        if (hasNumbers) return 'number';
        return 'string';
      }}

      function isValidDate(text) {{
        // More conservative date detection
        const date = new Date(text);
        return !isNaN(date.getTime()) &&
               (text.includes('-') || text.includes('/') || text.includes(':')) &&
               text.length > 4; // Avoid treating simple numbers as dates
      }}

      function isNumeric(text) {{
        if (!text || text.trim() === '') return false;

        // Handle common numeric formats including scientific notation
        const cleaned = text.replace(/[,%\s]/g, ''); // Remove commas, %, spaces

        // Check for valid number patterns
        const numberPattern = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
        return numberPattern.test(cleaned) && !isNaN(parseFloat(cleaned));
      }}

      function parseNumber(text) {{
        if (!text || text.trim() === '') return null;

        const cleaned = text.replace(/[,%\s]/g, ''); // Remove commas, %, spaces
        const num = parseFloat(cleaned);
        return isFinite(num) ? num : null;
      }}

      function compareCells(aCell, bCell, type) {{
        const aText = aCell?.textContent?.trim() ?? '';
        const bText = bCell?.textContent?.trim() ?? '';

        // Handle empty/null values
        if (!aText && !bText) return 0;
        if (!aText) return 1;  // Empty values go to end
        if (!bText) return -1;

        // Handle NaN values
        const aIsNaN = aText.toLowerCase() === 'nan';
        const bIsNaN = bText.toLowerCase() === 'nan';
        if (aIsNaN && bIsNaN) return 0;
        if (aIsNaN) return 1;  // NaN values go to end
        if (bIsNaN) return -1;

        if (type === 'number') {{
          const a = parseNumber(aText);
          const b = parseNumber(bText);

          if (a === null && b === null) return aText.localeCompare(bText);
          if (a === null) return 1;
          if (b === null) return -1;

          return a - b;
        }}

        if (type === 'date') {{
          const aDate = new Date(aText);
          const bDate = new Date(bText);

          if (isNaN(aDate.getTime()) && isNaN(bDate.getTime())) return aText.localeCompare(bText);
          if (isNaN(aDate.getTime())) return 1;
          if (isNaN(bDate.getTime())) return -1;

          return aDate.getTime() - bDate.getTime();
        }}

        // String comparison
        return aText.localeCompare(bText, undefined, {{ numeric: true, sensitivity: 'base' }});
      }}
    }})();
  </script>
</body>
</html>
"""
            table_path.write_text(table_doc, encoding="utf-8")
            table_iframes.append((table_rel, rel_csv))

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Benchmark Results</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --gap: 16px;
      --card-border: #d0d7de;
      --card-bg: #fff;
      --cols: 2;             /* user-selected (persisted) */
      --cols-applied: 2;     /* clamped to container width */
      --minColWidth: 520px;  /* prevent overly narrow plots */
      --globalFrameH: 720px; /* default global height */
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      margin: 2rem;
      background: #f6f8fa;
      color: #111;
    }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 1rem;
      align-items: end;
      margin-bottom: 1rem;
    }}
    h1 {{ margin: 0; font-size: 1.5rem; font-weight: 700; }}
    .subtitle {{ margin: 0; color: #57606a; font-size: 0.95rem; }}
    .controls {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
    .control {{
      display: inline-flex; align-items: center; gap: 0.5rem;
      background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 6px 10px;
    }}
    .control label {{ color: #57606a; font-size: 0.9rem; }}
    .btn {{
      font: inherit; border: 1px solid #d0d7de; background: #fff; border-radius: 6px;
      padding: 2px 8px; cursor: pointer; color: #111;
    }}
    .btn:hover {{ background: #f0f2f4; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(var(--cols-applied), minmax(var(--minColWidth), 1fr));
      gap: var(--gap);
    }}
    .tabs {{
      display: inline-flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }}
    .tab {{
      border: 1px solid #d0d7de;
      border-radius: 6px;
      background: #fff;
      color: #111;
      padding: 6px 14px;
      cursor: pointer;
      font: inherit;
    }}
    .tab:hover {{ background: #f0f2f4; }}
    .tab.active {{
      background: #0969da;
      border-color: #0969da;
      color: #fff;
    }}
    .tab:focus-visible {{
      outline: 2px solid #0969da;
      outline-offset: 2px;
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.active {{
      display: grid;
    }}
    .plot {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 0.5rem 0.5rem 0.25rem;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .plot iframe {{
      width: 100%;
      height: var(--globalFrameH);
      border: none;
      background: #fff;
      border-radius: 6px;
      display: block;
      overflow: hidden; /* we fully control height */
    }}
    .no-plots {{ margin-top: 1rem; color: #57606a; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Benchmark Results</h1>
      <p class="subtitle">Generated: {generated_ts}</p>
    </div>
    <div class="controls">
      <div class="control">
        <label for="cols">Columns</label>
        <select id="cols" aria-label="Select number of columns">
          <option value="1">1</option>
          <option value="2" selected>2</option>
          <option value="3">3</option>
          <option value="4">4</option>
        </select>
      </div>
      <div class="control">
        <label>Height</label>
        <button id="heightDec" class="btn" title="Decrease height">–</button>
        <span id="heightValue" aria-live="polite">720 px</span>
        <button id="heightInc" class="btn" title="Increase height">+</button>
      </div>
    </div>
  </header>
  <nav class="tabs" role="tablist">
    <button id="summary-tab" class="tab active" data-target="summary-panel" data-panel-key="summary" role="tab" aria-selected="true" aria-controls="summary-panel">Summary</button>
    <button id="slices-tab" class="tab" data-target="slices-panel" data-panel-key="slices" role="tab" aria-selected="false" aria-controls="slices-panel">Slice Plots</button>"""]

    # Only add Optuna tab if there are optuna htmls
    if optuna_htmls:
        parts.append("""    <button id="optuna-tab" class="tab" data-target="optuna-panel" data-panel-key="optuna" role="tab" aria-selected="false" aria-controls="optuna-panel">Optuna</button>""")

    parts.append("""    <button id="tables-tab" class="tab" data-target="tables-panel" data-panel-key="tables" role="tab" aria-selected="false" aria-controls="tables-panel">Tables</button>
  </nav>
  <div id="summary-panel" class="grid tab-panel active" data-panel="summary" role="tabpanel" aria-labelledby="summary-tab" aria-hidden="false">
""")

    if summary_htmls:
        for rel in summary_htmls:
            parts.append(f'''    <div class="plot" data-path="{rel}">
      <iframe class="plot-frame" src="{rel}" loading="lazy"></iframe>
    </div>''')
    else:
        parts.append('    <p class="no-plots">No summary plots were generated.</p>')

    parts.append("""  </div>
  <div id="slices-panel" class="grid tab-panel" data-panel="slices" role="tabpanel" aria-labelledby="slices-tab" aria-hidden="true">
""")

    if slice_htmls:
        for rel in slice_htmls:
            parts.append(f'''    <div class="plot" data-path="{rel}">
      <iframe class="plot-frame" src="{rel}" loading="lazy"></iframe>
    </div>''')
    else:
        parts.append('    <p class="no-plots">No slice plots were generated.</p>')

    parts.append("""  </div>""")

    # Only add Optuna panel if there are optuna htmls
    if optuna_htmls:
        parts.append("""  <div id="optuna-panel" class="grid tab-panel" data-panel="optuna" role="tabpanel" aria-labelledby="optuna-tab" aria-hidden="true">
""")
        for rel in optuna_htmls:
            parts.append(f'''    <div class="plot" data-path="{rel}">
      <iframe class="plot-frame" src="{rel}" loading="lazy"></iframe>
    </div>''')
        parts.append("""  </div>""")

    parts.append("""  <div id="tables-panel" class="grid tab-panel" data-panel="tables" role="tabpanel" aria-labelledby="tables-tab" aria-hidden="true">
""")

    if table_iframes:
        for iframe_src, original_rel in table_iframes:
            parts.append(f'''    <div class="plot" data-path="{original_rel}">
      <iframe class="plot-frame" src="{iframe_src}" loading="lazy" title="{original_rel}"></iframe>
    </div>''')
    else:
        parts.append('    <p class="no-plots">No tables were generated.</p>')

    parts.append("""  </div>
  <script>
    (function () {
      const root = document.documentElement;
      const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));
      const tabs = Array.from(document.querySelectorAll('.tab'));
      const selectCols = document.getElementById('cols');
      const decBtn = document.getElementById('heightDec');
      const incBtn = document.getElementById('heightInc');
      const heightValue = document.getElementById('heightValue');

      const LS_COLS_PREFIX = 'rl_bench_cols_';
      const LS_HEIGHT_PREFIX = 'rl_bench_height_';
      const DEFAULT_COLS = '2';
      const DEFAULT_HEIGHT = 720;

      const panelSettings = {};
      tabPanels.forEach(panel => {
        const key = panel.dataset.panel || panel.id || 'panel';
        const savedCols = localStorage.getItem(LS_COLS_PREFIX + key);
        const savedHeight = localStorage.getItem(LS_HEIGHT_PREFIX + key);
        panelSettings[key] = {
          cols: (savedCols && /^[1-4]$/.test(savedCols)) ? savedCols : DEFAULT_COLS,
          height: (savedHeight && /^[0-9]+$/.test(savedHeight)) ? parseInt(savedHeight, 10) : DEFAULT_HEIGHT,
        };
      });

      // Pull min column width from CSS var to avoid duplication.
      function parsePx(val) {
        const n = parseInt(String(val).replace(/[^0-9]/g, ''), 10);
        return Number.isNaN(n) ? null : n;
      }
      function getMinColWidth() {
        const v = getComputedStyle(root).getPropertyValue('--minColWidth');
        return parsePx(v) || 520;
      }

      const H_MIN = 360;
      const H_MAX = 2000;
      const H_STEP = 80;

      function getActivePanel() {
        return tabPanels.find(panel => panel.classList.contains('active')) || null;
      }

      function getActivePanelKey() {
        const active = getActivePanel();
        return active ? (active.dataset.panel || active.id || 'panel') : null;
      }

      function getCurrentHeight() {
        const key = getActivePanelKey();
        if (!key) return DEFAULT_HEIGHT;
        return panelSettings[key]?.height ?? DEFAULT_HEIGHT;
      }

      function setGlobalHeightPx(px) {
        const key = getActivePanelKey();
        if (!key) return;
        const clamped = Math.max(H_MIN, Math.min(px, H_MAX));
        root.style.setProperty('--globalFrameH', String(clamped) + 'px');
        heightValue.textContent = String(clamped) + ' px';
        if (!panelSettings[key]) {
          panelSettings[key] = { cols: DEFAULT_COLS, height: DEFAULT_HEIGHT };
        }
        panelSettings[key].height = clamped;
        localStorage.setItem(LS_HEIGHT_PREFIX + key, String(clamped));
      }

      function clampAndApplyColumns() {
        const requested = parseInt(getComputedStyle(root).getPropertyValue('--cols'), 10) || 2;
        const active = getActivePanel();
        if (!active) {
          root.style.setProperty('--cols-applied', String(requested));
          return;
        }
        let gridWidth = active.getBoundingClientRect().width;
        if (!gridWidth) gridWidth = active.clientWidth;
        if (!gridWidth) gridWidth = root.clientWidth;
        if (!gridWidth) gridWidth = window.innerWidth || 1200;
        const maxCols = Math.max(1, Math.floor(gridWidth / getMinColWidth()));
        root.style.setProperty('--cols-applied', String(Math.min(requested, maxCols)));
      }

      function applyPanelSettings(key) {
        if (!key) return;
        if (!panelSettings[key]) {
          panelSettings[key] = { cols: DEFAULT_COLS, height: DEFAULT_HEIGHT };
        }
        const settings = panelSettings[key];
        root.style.setProperty('--cols', settings.cols);
        selectCols.value = settings.cols;
        root.style.setProperty('--globalFrameH', String(settings.height) + 'px');
        heightValue.textContent = String(settings.height) + ' px';
        clampAndApplyColumns();
      }

      // ----- Init: tabs -----
      tabPanels.forEach(panel => {
        const isActive = panel.classList.contains('active');
        panel.setAttribute('aria-hidden', String(!isActive));
      });
      const activeKeyAtInit = getActivePanelKey();
      if (activeKeyAtInit) {
        applyPanelSettings(activeKeyAtInit);
      } else if (tabPanels.length) {
        const fallbackKey = tabPanels[0].dataset.panel || tabPanels[0].id || 'panel';
        applyPanelSettings(fallbackKey);
      }

      tabs.forEach(tab => {
        tab.setAttribute('aria-selected', String(tab.classList.contains('active')));
        tab.addEventListener('click', () => {
          if (tab.classList.contains('active')) return;
          tabs.forEach(t => {
            const active = t === tab;
            t.classList.toggle('active', active);
            t.setAttribute('aria-selected', String(active));
          });
          const targetId = tab.dataset.target;
          tabPanels.forEach(panel => {
            const active = panel.id === targetId;
            panel.classList.toggle('active', active);
            panel.setAttribute('aria-hidden', String(!active));
          });
          const panelEl = targetId ? document.getElementById(targetId) : null;
          const key = panelEl ? (panelEl.dataset.panel || panelEl.id || 'panel') : getActivePanelKey();
          if (key) {
            applyPanelSettings(key);
          }
        });
      });

      // ----- Wire up controls -----
      selectCols.addEventListener('change', (e) => {
        const val = String(e.target.value);
        if (/^[1-4]$/.test(val)) {
          const key = getActivePanelKey();
          if (!key) return;
          root.style.setProperty('--cols', val);
          panelSettings[key].cols = val;
          localStorage.setItem(LS_COLS_PREFIX + key, val);
          clampAndApplyColumns();
        }
      });

      decBtn.addEventListener('click', () => setGlobalHeightPx(getCurrentHeight() - H_STEP));
      incBtn.addEventListener('click', () => setGlobalHeightPx(getCurrentHeight() + H_STEP));

      // Initial clamp; re-clamp on resize
      clampAndApplyColumns();
      window.addEventListener('resize', clampAndApplyColumns);
    })();
  </script>
</body>
</html>""")

    html = "\n".join(parts)
    index_path.write_text(html, encoding="utf-8")
    logger.info(f"Generated HTML index: {index_path}")
