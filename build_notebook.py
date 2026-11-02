"""Genera Lab9_Series_de_Tiempo.ipynb: contiene TODO el código (vía %%writefile) y los resultados."""
import nbformat as nbf
from pathlib import Path

R = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))
src = lambda p: code(f"%%writefile {p}\n" + (R / p).read_text())
doc = lambda p: md((R / p).read_text()) if (R / p).exists() else None

md("""# CC3092 · Laboratorio 9 — Series de tiempo y forecasting
**Autores:** Javier López (23415) · Javier Cifuentes (23079)
**Dataset:** ElectricityLoadDiagrams20112014 (UCI) · **Objetivo:** consumo horario (kWh) por cliente, pronóstico day-ahead H=24 emitido a las 00:00.

El notebook es autocontenido: cada celda `%%writefile` crea el módulo correspondiente en `src/` o el script `run_*.py`, y las celdas de ejecución los corren.
Con `RUN_TRAINING = False` (por defecto) el notebook **carga los resultados ya calculados** en `results/`; con `True` vuelve a entrenar todo (≈1.5–2 h en CPU de 2 núcleos; mucho menos con GPU).

**Hardware de los resultados reportados:** CPU de 2 núcleos, sin GPU (el código usa CUDA automáticamente si existe).
**Restricción de cómputo:** se usan 100 clientes (mínimo permitido) elegidos con semilla 42 entre los 317 con datos completos; el mismo panel para todos los modelos.""")
code("""import os, sys, json, glob, subprocess
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
RUN_TRAINING = False   # True = reentrenar todo
os.makedirs('src', exist_ok=True); os.makedirs('docs', exist_ok=True)
def run(cmd):
    if RUN_TRAINING:
        print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout[-3000:])
    else:
        print('(RUN_TRAINING=False) se usan resultados guardados para:', cmd)""")

md("## 0. Configuración y utilidades\nRutas, fechas de partición, semilla fija (42) y selección automática de dispositivo (CUDA si está disponible, si no CPU).")
code("open('src/__init__.py', 'a').close()")
src("src/config.py")

md("## 1–2. Datos, exploración y preparación\n"
   "Pipeline: crudo 15 min (kW) → horario (kWh) → corrección de cambios de horario → filtro de clientes completos → muestra de 100 clientes → normalización con estadísticos de entrenamiento → covariables de calendario → ventanas.")
src("src/data.py")
src("src/prep.py")
md("### Dataset de ventanas deslizantes (L=168, H=24, stride=1)")
src("src/windows.py")
md("### Exploración, conteo de ventanas y auditoría de fugas\nEl script contiene `assert`s: si alguna fuga existiera, la celda fallaría.")
src("run_00_eda.py")
code("%run run_00_eda.py")
code("for f in ['perfiles','serie_diaria_feriados','navidad','escalas']: display(Image(f'results/figures/{f}.png'))")
doc("docs/respuestas_seccion2.md")

doc("docs/investigacion.md")

md("## 4. Protocolo de validación y métricas\n`backtest(modelo)` es la única función de evaluación. Corta `Y[:, :o]` antes de llamar a `modelo.predict`, de modo que ningún modelo puede ver datos posteriores a las 23:00 del día previo. Las librerías (StatsForecast/NeuralForecast) entran por un adaptador que verifica `cutoff == o-1`.")
src("src/metrics.py")
src("src/backtest.py")
src("src/store.py")

md("## 5. Modelos\n### 5.a Baselines (StatsForecast)")
src("src/baselines.py")
src("run_02_baselines.py")
code("run('python run_02_baselines.py')")
md("### 5.b LightGBM global, 5.c modelos propios en PyTorch (MLP, DLinear, NLinear, LSTM, Seq2Seq) y Chronos-Bolt")
src("src/lgbm.py")
src("src/models_torch.py")
src("src/train.py")
src("src/chronos_model.py")
src("run_03_models.py")
code("run('python run_03_models.py lgbm linear chronos lstm')")
md("### 5.d NeuralForecast: N-HiTS, TCN, PatchTST, iTransformer y DeepAR")
src("run_04_nf.py")
code("run('python run_04_nf.py')")
md("### 5.1–5.3 Experimentos: iteraciones de hiperparámetros, historia, local vs global, semillas, reentrenamiento mensual y H=168")
src("select_best.py")
src("run_05_experiments.py")
code("run('python run_05_experiments.py hp hist'); run('python select_best.py'); run('python run_05_experiments.py seeds local retrain h168')")

md("## 6. Comparación de resultados")
src("run_06_report.py")
code("%run run_06_report.py")
code("""tab = pd.read_csv('results/tabla_comparativa.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
display(tab.round(4))""")
code("for f in ['pareto','mae_por_paso','mase_vs_historia','estrategias_lstm','mae_por_hora','curvas_perdida','bandas_DeepAR','bandas_Chronos-Bolt']:\n    p=f'results/figures/{f}.png'\n    if os.path.exists(p): display(Image(p))")
md("### Tabla de iteraciones (selección por validación)")
code("it = pd.read_csv('results/tabla_iteraciones.csv'); display(it)")
md("### Local vs global, semillas y reentrenamiento")
code("""for f in ['results/experiments/local_vs_global.csv','results/experiments/seeds.csv']:
    if os.path.exists(f): display(pd.read_csv(f).round(4))
for f in ['results/experiments/retrain_dl.json','results/experiments/best_own.json']:
    if os.path.exists(f): print(f, json.load(open(f)))""")

doc("docs/discusion.md")
md("## Entregables\n- `predicciones_test.csv.gz` (`unique_id, ds, modelo, y_hat`) en `results/`.\n- Reporte PDF en `reporte/`.\n- `requirements.txt` y `README.md` para reproducir.")
nb["cells"] = [c for c in cells if c is not None]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, R / "Lab9_Series_de_Tiempo.ipynb")
print("ok", len(nb.cells))
