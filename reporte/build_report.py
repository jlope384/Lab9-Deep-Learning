"""reporte/reporte.md (+ tablas/figuras de results/) -> reporte/Reporte_Lab9.pdf (Playwright/Chromium)."""
import base64, json, re
from pathlib import Path
import markdown, pandas as pd

R = Path(__file__).resolve().parents[1]
OUT = R / "reporte"
md_text = (OUT / "reporte.md").read_text()


def img(name, width="100%"):
    p = R / "results" / "figures" / f"{name}.png"
    if not p.exists():
        return f"<p><i>[figura {name} no disponible]</i></p>"
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def table_comp():
    t = pd.read_csv(R / "results" / "tabla_comparativa.csv")
    t = t.rename(columns={"name": "Modelo", "family": "Familia", "strategy": "Estrategia", "covariates": "Cov.",
                          "params_total": "Params", "train_s": "Entren. s", "infer_s": "Infer. s",
                          "MASE_H168": "MASE H168", "MASE_xmas": "MASE Nav.", "coverage_80": "Cob. P10–P90",
                          "pinball": "Pinball", "mejora_vs_SN24_%": "Δ vs SN24 %"})
    t["Params"] = t["Params"].map(lambda v: "-" if pd.isna(v) or v == 0 else (f"{v/1e6:.2f}M" if v >= 1e6 else f"{int(v):,}"))
    t["Cov."] = t["Cov."].map(lambda s: "sí" if isinstance(s, str) and s != "no" else "no")
    cols = ["Modelo", "Familia", "L", "Estrategia", "Cov.", "Params", "Entren. s", "Infer. s", "MAE", "RMSE",
            "WAPE", "MASE", "MASE H168", "MASE Nav.", "Pinball", "Cob. P10–P90", "Δ vs SN24 %"]
    fmt = {"Entren. s": "{:.0f}", "Infer. s": "{:.1f}", "MAE": "{:.1f}", "RMSE": "{:.1f}", "WAPE": "{:.3f}",
           "MASE": "{:.3f}", "MASE H168": "{:.3f}", "MASE Nav.": "{:.3f}", "Pinball": "{:.1f}",
           "Cob. P10–P90": "{:.2f}", "Δ vs SN24 %": "{:+.1f}"}
    t = t[cols].copy()
    for c, f in fmt.items():
        t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f.format(v))
    t["L"] = t["L"].map(lambda v: "-" if pd.isna(v) else str(int(v)))
    return t.to_html(index=False, classes="tbl", border=0)


def table_iter():
    p = R / "results" / "tabla_iteraciones.csv"
    if not p.exists():
        return ""
    t = pd.read_csv(p)
    t["cambio"] = t.tag.str.replace("hp_", "", regex=False)
    t = t[["cambio", "L", "norm", "lr", "batch", "params", "epochs", "epoch_s", "train_s", "infer_s", "peak_mem_MB", "val_MASE", "test_MASE"]]
    t = t.rename(columns={"peak_mem_MB": "mem MB*", "epoch_s": "s/epoch", "train_s": "train s", "infer_s": "infer s"})
    return t.to_html(index=False, classes="tbl small", border=0)


def csv_table(path, cols=None, fmt="{:.3f}"):
    p = R / path
    if not p.exists():
        return ""
    t = pd.read_csv(p)
    if cols: t = t[cols]
    return t.to_html(index=False, classes="tbl small", border=0, float_format=lambda v: fmt.format(v))


import os
disc = (R / "docs" / "discusion.md").read_text().replace("## 8. Discusión y análisis (con base en los resultados de prueba)", "## 5. Discusión y conclusiones")
repl = {
    "{{DISCUSION}}": disc,
    "{{REPO}}": os.environ.get("REPO_URL", "[agregar enlace al repositorio de GitHub]"),
    "{{TABLA_COMPARATIVA}}": table_comp(),
    "{{TABLA_ITERACIONES}}": table_iter(),
    "{{TABLA_LOCAL}}": csv_table("results/experiments/local_vs_global.csv", ["unique_id", "mean_kwh", "MASE_local", "MASE_global"], "{:.2f}"),
}
for k, v in repl.items():
    md_text = md_text.replace(k, v)
md_text = re.sub(r"\{\{FIG:([\w\-]+)(?::(\d+%))?\}\}", lambda m: img(m.group(1), m.group(2) or "100%"), md_text)
body = markdown.markdown(md_text, extensions=["tables"])
css = """
@page { size: Letter; margin: 13mm 12mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8.3pt; line-height: 1.28; color: #111; }
h1 { font-size: 13pt; margin: 0 0 2px; } h2 { font-size: 10pt; margin: 7px 0 2px; border-bottom: 1px solid #999; }
h3 { font-size: 8.8pt; margin: 5px 0 1px; } p, li { margin: 2px 0; } ul { margin: 2px 0; padding-left: 16px; }
table.tbl { border-collapse: collapse; width: 100%; font-size: 6.6pt; margin: 3px 0; }
table.tbl th { background: #e8e8e8; } table.tbl td, table.tbl th { padding: 1px 3px; border-bottom: 1px solid #ddd; text-align: right; }
table.tbl td:first-child, table.tbl th:first-child { text-align: left; }
table.small { font-size: 6.3pt; }
table:not(.tbl) { border-collapse: collapse; font-size: 7.4pt; } table:not(.tbl) td, table:not(.tbl) th { border: 1px solid #ccc; padding: 1px 4px; }
.row { display: flex; gap: 6px; } .row > div { flex: 1; } img { display: block; margin: 2px auto; }
.cap { font-size: 7pt; color: #444; text-align: center; margin-top: 0; }
"""
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
(OUT / "reporte.html").write_text(html)
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.set_content(html); pg.wait_for_timeout(300)
    pg.pdf(path=str(OUT / "Reporte_Lab9.pdf"), format="Letter", print_background=True,
           margin=dict(top="13mm", bottom="13mm", left="12mm", right="12mm")); b.close()
from pypdf import PdfReader
print("páginas:", len(PdfReader(OUT / "Reporte_Lab9.pdf").pages))
