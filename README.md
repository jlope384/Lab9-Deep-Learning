# CC3092 · Laboratorio 9 — Series de tiempo y forecasting
**Autores:** Javier López (23415) · Javier Cifuentes (23079)

Pronóstico day-ahead (H = 24 h, emitido a las 00:00) del consumo eléctrico horario (kWh) de clientes de Portugal
(UCI *ElectricityLoadDiagrams20112014*). Baselines estadísticos, LightGBM, modelos propios en PyTorch,
NeuralForecast y Chronos-Bolt zero-shot, todos evaluados con **una sola** función `backtest()` sobre los 92
orígenes diarios de prueba (oct–dic 2014).

## Estructura
```
Lab9/
├── Lab9_Series_de_Tiempo.ipynb   # notebook completo (contiene todo el código vía %%writefile + resultados)
├── src/
│   ├── config.py        # rutas, fechas de split, semillas, dispositivo (CUDA/CPU)
│   ├── data.py          # descarga, 15min->kWh horario, cambios de horario, filtro, feriados, calendario, Scaler
│   ├── prep.py          # contexto común (mismo panel para todos los modelos)
│   ├── windows.py       # Dataset PyTorch de ventanas deslizantes (L, H, stride)
│   ├── metrics.py       # MAE, RMSE, WAPE, MAPE, MASE, pinball, cobertura
│   ├── backtest.py      # backtest(modelo) con origen móvil + adaptador para librerías
│   ├── store.py         # guarda predicciones/métricas por modelo
│   ├── baselines.py     # StatsForecast: Naive, SeasonalNaive 24/168, MSTL
│   ├── lgbm.py          # LightGBM global (rezagos 1–24, 168 + calendario), recursivo
│   ├── models_torch.py  # MLP, DLinear, NLinear, LSTM (AR / multi-salida), Seq2Seq, RevIN
│   ├── train.py         # entrenamiento, early stopping en validación, registro por epoch
│   └── chronos_model.py # Chronos-Bolt zero-shot (predict_quantiles)
├── run_00_eda.py              # exploración + conteo de ventanas + auditoría de fugas (asserts)
├── run_02_baselines.py        # baselines (H=24 y H=168)
├── run_03_models.py           # LightGBM, lineales, MLP, Chronos, LSTMs, Seq2Seq
├── run_04_nf.py               # N-HiTS, TCN, PatchTST, iTransformer, DeepAR (NeuralForecast)
├── run_05_experiments.py      # iteraciones HP, historia, local vs global, semillas, reentrenamiento, H=168
├── select_best.py             # mejor modelo propio por MASE de VALIDACIÓN
├── run_06_report.py           # tabla comparativa, gráficas, predicciones
├── build_notebook.py          # genera el notebook
├── docs/                      # investigación (sección 3), respuestas y discusión
├── reporte/                   # reporte PDF (≤ 5 páginas) y su fuente
└── results/
    ├── tabla_comparativa.csv, tabla_iteraciones.csv, eda_summary.json
    ├── predicciones_test.csv.gz     # unique_id, ds, modelo, y_hat
    ├── bt/            # métricas por modelo: global, por cliente, por paso, por hora
    ├── experiments/   # iteraciones, historia, local/global, semillas, reentrenamiento
    ├── train_logs/    # historial por epoch de cada modelo propio
    ├── models/        # pesos .pt
    └── figures/
```

## Reproducir
```bash
pip install -r requirements.txt          # para GPU: instalar torch con CUDA primero
python run_00_eda.py                     # descarga el dataset (~250 MB) la primera vez
python run_02_baselines.py
python run_03_models.py lgbm linear chronos lstm
python run_04_nf.py
python run_05_experiments.py hp hist && python select_best.py
python run_05_experiments.py seeds local retrain h168
python run_04_nf.py PatchTST hist=3 && python run_04_nf.py PatchTST hist=12
python run_06_report.py
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace Lab9_Series_de_Tiempo.ipynb
python reporte/build_report.py
```
O bien abrir el notebook y poner `RUN_TRAINING = True`.

## Decisiones principales
- **Panel:** 317 clientes con datos completos desde 2012-01-01; se usan **100** (semilla 42) por restricción de cómputo (CPU de 2 núcleos, sin GPU), los mismos para todos los modelos.
- **Unidades:** kWh horario = Σ(4 registros de kW de 15 min) / 4. La marca de tiempo es el inicio del intervalo.
- **Cambio de horario:** marzo (hora 01:00 en cero) → interpolación; octubre (hora 01:00 con 2 h) → /2.
- **Normalización:** z-score por serie con estadísticos de entrenamiento (≤ 2014-06-30).
- **Splits:** entrenamiento 2012-01-01..2014-06-30 · validación 2014-07-01..2014-09-30 (early stopping y selección) · prueba 2014-10-01..2014-12-31 (92 orígenes).
- **MASE:** escala = MAE in-sample del seasonal naive (m = 24) en entrenamiento, por serie; promedio entre series.
- **Presupuesto de cómputo:** los modelos de deep learning se entrenaron en CPU con un número limitado de ventanas por epoch y de pasos (documentado en cada log); con GPU se pueden aumentar en `run_03/04/05`.

## Checklist de requisitos del PDF
| Sección | Requisito | Dónde | Estado |
|---|---|---|---|
| 1 | Descarga, kWh horario, 2012–2014, clientes completos, subconjunto ≥100 con semilla | `src/data.py`, `run_00_eda.py` | ✅ 317 → 100 (semilla 42) |
| 2 | Nº de clientes y observaciones, cambio de horario, perfiles hora/día + feriados, Navidad/Año Nuevo, escalas, normalización, covariables | `docs/respuestas_seccion2.md`, `results/figures/` | ✅ |
| 2 | Dataset PyTorch de ventanas (L=168, H=24, stride 1) y conteo train/val/test | `src/windows.py` | ✅ 2.17 M / 218.5 k / 218.5 k |
| 2 | Auditoría de las 4 fugas con código | `run_00_eda.py` (asserts) | ✅ |
| 3 | Investigación de LSTM/GRU, Conv1d causal, Transformer, Nixtla, Chronos y conceptos | `docs/investigacion.md` | ✅ |
| 4 | `backtest(modelo)` con 92 orígenes, pesos entrenados una vez | `src/backtest.py` | ✅ |
| 4 | Reentrenamiento mensual (mejor DL y mejor baseline) | `run_05_experiments.py retrain` | ✅ |
| 4 | MAE, RMSE (también por cliente), WAPE, MAPE, MASE; por paso, por hora y semana 22–28 dic | `src/metrics.py`, `results/bt/` | ✅ |
| 5 | Naive, SN24, SN168, MSTL · LightGBM · MLP, DLinear, NLinear · LSTM · Seq2Seq · N-HiTS, TCN · PatchTST, iTransformer · DeepAR · Chronos-Bolt | `run_02/03/04` | ✅ (ver limitaciones) |
| 5.1 | Estrategias autorregresiva / multi-salida / encoder-decoder con LSTM | `results/figures/estrategias_lstm.png` | ✅ |
| 5.2 | Iteraciones: L (48/168/336), historia (3/12/30 meses), normalización, oculto, capas, dropout, lr y batch, con todos los registros pedidos | `results/tabla_iteraciones.csv`, `results/experiments/` | ✅ |
| 5.3 | Local vs global en 10 clientes | `results/experiments/local_vs_global.csv` | ✅ |
| 6 | Tabla comparativa ordenada por MASE (incluye MASE H=168, Navidad, pinball y cobertura) + 3 gráficas | `results/tabla_comparativa.csv`, `results/figures/` | ✅ |
| 8 | Discusión y 3 semillas | `docs/discusion.md`, `results/experiments/seeds.csv` | ✅ |
| Entregables | PDF ≤5 páginas, notebook, archivo de predicciones `unique_id, ds, modelo, y_hat` | `reporte/Reporte_Lab9.pdf`, notebook, `results/predicciones_test.csv.gz` | ✅ |

### Limitaciones (cómputo: CPU de 2 núcleos, 8 GB RAM, sin GPU)
- 100 clientes (el mínimo permitido).
- Modelos propios: muestreo de 8 k–40 k ventanas por epoch; las LSTM quedan sub-entrenadas (se incluye `LSTM-MO-long` con más presupuesto).
- NeuralForecast: 150–200 pasos y **12 meses** de historia (con 30 meses se quedaba sin memoria); sin early stopping interno (presupuesto fijo de pasos). N-HiTS, TCN y DeepAR usan el z-score por serie del pipeline y `scaler_type="identity"`, porque el escalado por ventana de NF también normaliza las covariables futuras y diverge con `mes`/`feriado`, que son casi constantes dentro de la ventana.
- MSTL de StatsForecast cuesta ~20–35 s por origen en esta CPU: se ajusta en cada origen con las 2 semanas previas.
- Con GPU basta con aumentar `epochs`, `samples_per_epoch` y `max_steps` en los scripts.
