### Respuestas — sección 2 (valores calculados por `run_00_eda.py`)

**Unidades.** Cada registro es la potencia media (kW) de un intervalo de 15 min, así que su energía es kW × 0.25 h. El consumo horario es `kWh = (x₁+x₂+x₃+x₄)/4`. Al revisar los días de cambio de horario se confirmó que la marca de tiempo corresponde al **inicio** del intervalo (el bloque anómalo de marzo y octubre está en 01:00–01:45), por lo que se agrupa con `floor` a la hora.

**Clientes y observaciones.** El archivo crudo tiene 140,256 × 370 registros. Quedan **317 clientes** con datos completos desde 2012-01-01: tienen consumo > 0 el primer día, no tienen NaN y ningún tramo de ceros ≥ 7 días (53 se excluyen porque empiezan tarde o se apagan). Cada serie tiene **26,304 horas** (1,096 días × 24) y el panel filtrado suma **8,338,368** observaciones. Por restricción de cómputo (CPU de 2 núcleos, sin GPU) se usan **100 clientes** elegidos con `numpy.random.default_rng(42)` (2,630,400 observaciones), los mismos para todos los modelos.

**Cambios de horario.** Todos los días tienen 96 registros. En marzo (día de 23 h) la hora 01:00 viene en cero: la suma del panel en esa hora es ~4.3k kWh, contra ~120k en las horas vecinas. Se imputa con la media de 00:00 y 02:00 de cada cliente. En octubre (día de 25 h) la hora 01:00 acumula dos horas: su proporción frente a la hora previa es de 1.54 a 1.70, cuando lo normal es menos de 1, así que se divide entre 2. Así la grilla queda regular, con 24 h/día y sin huecos.

**Perfiles y feriados.** El consumo relativo mínimo es de 0.62× la media hacia las 04:00 y el máximo de 1.29× a las 19–20 h. Hay una meseta de 11 a 17 h. La diferencia entre días de la semana es pequeña en el promedio normalizado (domingo ≈ 0.99, viernes ≈ 1.02) porque el panel mezcla perfiles comerciales y residenciales. Los feriados se obtienen con `holidays.Portugal`, que respeta la suspensión de Corpo de Deus, 5-oct, 1-nov y 1-dic en 2013–2014.

**Navidad y Año Nuevo.** Frente al mismo día de la semana de las 4 semanas previas, el 25-dic cae a 0.74 (2012), 0.76 (2013) y **0.67 (2014, en prueba)**. El 1-ene cae a 0.77–0.78. En cambio, el 24-dic (0.96–0.98) y el 31-dic (0.97–1.02) casi no cambian. Es un cambio de régimen corto y fuerte, concentrado en los días feriados.

**Escalas.** El consumo medio horario en entrenamiento va de **4.3 a 48,818 kWh** (mediana 148) en los 317 clientes, y de **8.3 a 11,480 kWh** (mediana 154) en los 100 usados: más de 3 órdenes de magnitud. Se elige **z-score por serie con media y desviación del bloque de entrenamiento**. Una normalización global haría que los clientes grandes dominaran la pérdida, y una por ventana (RevIN) pierde la información de nivel. Ambas alternativas se comparan en las iteraciones (sección 5.2).

**Covariables de calendario conocidas a futuro.** Son la hora (seno/coseno, periodo 24), el día de la semana (seno/coseno, periodo 7), el mes (seno/coseno, periodo 12), `is_holiday` y `is_weekend`: 8 en total. Seno/coseno conserva la continuidad circular (23 h está junto a 0 h) con 2 columnas en vez de 24 one-hot. Como son funciones deterministas de la fecha, se conocen en el instante del pronóstico.

**Ventanas (L=168, H=24, stride=1).** Una ventana pertenece a la partición donde cae su objetivo completo. La entrada puede usar la historia del bloque anterior, que ya es conocida en el origen.

| partición | ventanas por serie | ventanas del panel (100 series) |
|---|---|---|
| entrenamiento | 21,697 | 2,169,700 |
| validación | 2,185 | 218,500 |
| prueba | 2,185 | 218,500 (backtest: 92 orígenes diarios × 100) |

Los tensores tienen forma `X: (batch, 168, 9)` (y normalizada + 8 covariables), `X_fut: (batch, 24, 8)` y `y: (batch, 24)`. Por costo de CPU, cada epoch muestrea con reemplazo un subconjunto fijo de ventanas de entrenamiento (`RandomSampler(num_samples)`).

**Auditoría de fugas** (todas verificadas con `assert`):
1. *Split aleatorio de ventanas:* las particiones son temporales. Los objetivos de entrenamiento terminan el 2014-06-30 23:00, los de validación cubren del 2014-07-01 al 2014-09-30 y los de prueba empiezan el 2014-10-01. Solo se barajan ventanas dentro de entrenamiento.
2. *Escalar con toda la serie:* μ y σ salen de `Y[:, :val_pos]`. Multiplicar validación y prueba por 10 no cambia el scaler. Usar la serie completa habría desplazado μ en 5.17 kWh en promedio.
3. *Early stopping con la prueba:* `train_model` solo evalúa en los 92 orígenes de validación (último objetivo: 2014-09-30 23:00). La prueba se usa una sola vez, en `backtest(split='test')`.
4. *Covariables mal alineadas:* `X_fut` es exactamente el calendario de `[t, t+H)`. Poner `y[t:] = 999` no altera la entrada `X` ni las filas de rezagos de LightGBM. Además, `backtest()` entrega `Y[:, :o]` a cada modelo, y en las librerías se verifica `cutoff == o − 1`.
