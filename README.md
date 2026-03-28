# F1 Winner Predictor 2026

Predictor del ganador de cada Gran Premio de Fórmula 1 basado en datos de clasificación (sábado). Combina dos modelos de machine learning — **XGBoost** (desplegado en AWS Lambda) y **TabNet** (inferencia local) — y sincroniza predicciones y métricas con Google Sheets para visualización en Looker Studio.

---

## Arquitectura general

```
FastF1 API
    │
    ▼
data_collection.py  ──►  feature_engineering.py
                                   │
          ┌────────────────────────┤
          │                        │
    XGBoost (train.py)      TabNet (train.py)
          │                        │
          ▼                        ▼
    S3 (models/)          models/ (local)
          │
          ▼
    AWS Lambda                predict.py (local)
    (predict_lambda.py)            │
          │                        │
          └──────────┬─────────────┘
                     ▼
            S3: predictions/history.csv
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Google Sheets           Athena
    (Looker Studio       (disponible para
     — fuente activa)    consultas SQL ad-hoc)
```

---

## Estructura del proyecto

```
f1-winner-predictor/
├── config.py                  # Configuración central (paths, features, parámetros)
├── train.py                   # Entrenamiento XGBoost + TabNet con Optuna
├── predict.py                 # Inferencia TabNet local + registro de resultados
├── predict_proba_all.py       # Probabilidades de todos los pilotos (XGBoost + TabNet, sin S3)
├── requirements.txt           # Dependencias del entorno local/Docker
├── requirements-lambda.txt    # Dependencias del contenedor Lambda
├── Dockerfile                 # Imagen para entrenamiento y predicción local
├── Dockerfile.lambda          # Imagen para AWS Lambda (ECR)
├── Dockerfile.layer           # Layer de dependencias Lambda
├── src/
│   ├── __init__.py
│   ├── data_collection.py     # Descarga datos de FastF1 (quali, carrera, meteorología)
│   ├── feature_engineering.py # Construcción de features para XGBoost y TabNet
│   ├── aws_utils.py           # S3, Google Sheets sync (Athena configurado para SQL ad-hoc)
│   ├── predict_lambda.py      # Handler de AWS Lambda (inferencia XGBoost)
│   └── circuit_metadata.py    # Tabla estática de metadatos de circuitos
└── models/                    # Artefactos locales (gitignored, gestionados vía S3)
    ├── xgboost_f1_winner.pkl
    ├── label_encoders.pkl
    ├── tabnet_model.zip
    └── scaler.pkl
```

---

## Features utilizadas (28 variables)

| Categoría | Feature | Descripción |
|---|---|---|
| **Clasificación** | `grid_position` | Posición en parrilla (sábado, no penalizaciones) |
| **Clasificación** | `quali_gap_to_pole_s` | Gap al pole en segundos. Se calcula como `min(Q1, Q2, Q3)` para capturar la mejor vuelta real de cada piloto independientemente de la sesión en que fue eliminado |
| **Práctica libre** | `fp2_long_run_pace_gap_s` | Gap al mejor ritmo de carrera en FP2 |
| **Práctica libre** | `fp3_gap_to_best_s` | Gap al mejor tiempo en FP3 |
| **Sprint Race** | `sprint_race_position` | Posición en el Sprint Race (0 = GP sin sprint). Proxy de ritmo real en carrera, disponible antes de la clasificación del sábado |
| **Meteorología** | `rain_probability` | Fracción de la sesión con lluvia (0–1) |
| **Meteorología** | `is_wet_qualifying` | Binario: clasificación en mojado |
| **Meteorología** | `track_temp_c`, `humidity_pct`, `wind_speed_ms` | Condiciones durante la clasificación |
| **Campeonato** | `driver_champ_pos`, `driver_champ_points` | Posición y puntos del piloto antes de la carrera |
| **Campeonato** | `constructor_champ_pos`, `constructor_champ_points` | Posición y puntos del constructor |
| **Historial piloto** | `driver_avg_finish_l3` | Media de posición final en las últimas 3 carreras |
| **Historial piloto** | `driver_win_rate_l5` | Tasa de victorias en las últimas 5 carreras |
| **Historial piloto** | `driver_wet_win_rate` | Tasa de victorias en carreras mojadas (últimas 10) |
| **Historial piloto** | `driver_current_season_win_rate` | Tasa de victorias en la temporada actual. Evita sesgo por dominancia histórica de un piloto en temporadas anteriores |
| **Historial piloto** | `driver_races_since_last_win` | Carreras desde la última victoria (máximo 50). Penaliza a pilotos que no ganan hace tiempo |
| **Historial circuito** | `driver_best_finish_circuit`, `driver_avg_finish_circuit` | Mejor y media de resultado en este circuito en los **últimos 3 años** (se excluye el histórico más antiguo para evitar que victorias obsoletas distorsionen la predicción) |
| **Circuito (FastF1)** | `track_length_km`, `corner_count` | Longitud y número de curvas (dinámico) |
| **Circuito (estático)** | `overtake_difficulty`, `drs_zones`, `avg_safety_car_prob` | Dificultad de adelantamiento, zonas DRS, probabilidad histórica de SC |
| **Contexto** | `race_number` | Número de ronda en la temporada |
| **Codificados** | `circuit_encoded`, `constructor_encoded` | Label encoding de circuito y constructor |

---

## Modelos

### XGBoost (nube — AWS Lambda)
- Clasificador binario (`is_winner = 1` para el ganador de cada carrera)
- `scale_pos_weight = 5`: corrección parcial del desbalance. El valor completo (21) sobreconcentra las probabilidades en el favorito; con 5 el modelo mantiene spread real entre los pilotos de cabeza.
- `base_score = 0.045` (= 1/22): prior natural de victoria. El default de XGBoost (0.5) inflaba artificialmente las estimaciones de pilotos con pocos datos.
- Hiperparámetros optimizados con **Optuna** (ROC-AUC, 5-fold CV estratificado)
- Entrenado con datos 2023–2026 con **pesos de recencia** `{2023:1, 2024:1, 2025:2, 2026:2}` mediante `sample_weight`
- Artefactos en S3: `models/xgboost_f1_winner.pkl` + `models/label_encoders.pkl` + `models/xgb_temperature.pkl`

#### Calibración de probabilidades (Temperature Scaling)

XGBoost predice cada piloto como un clasificador binario independiente. Esto genera dos problemas:

1. **Confianza extrema**: sin calibración, el poleman puede recibir ~90–99%, muy por encima de la frecuencia real de victorias desde pole (~40–45%).
2. **No suman 1**: las probabilidades de los 22 clasificadores no tienen por qué sumar 100%.

**Temperature Scaling** (`train.py → _fit_temperature`): XGBoost se entrena con el **80%** de los datos. Con el **20% restante** se minimiza la NLL para encontrar el $T$ óptimo, que se guarda en `models/xgb_temperature.pkl`.

Al predecir (`predict_lambda.py`, `predict.py`, `predict_proba_all.py`), se extraen los logits ($z = \log\frac{p}{1-p}$), se dividen entre $T$, se pasa por sigmoid y finalmente se normaliza entre los $N$ pilotos:

$$p_{\text{cal}} = \frac{1}{1 + e^{-z/T}}, \quad p_{\text{norm}} = \frac{p_{\text{cal}}}{\sum_i p_{\text{cal},i}}$$

### TabNet (local)
- Red neuronal tabular con mecanismo de atención (pytorch-tabnet)
- Mismos datos y features que XGBoost
- Estandarización previa con `StandardScaler`
- Hiperparámetros optimizados con **Optuna** (ROC-AUC, validación 20%)
- Entrenado con **oversampling por recencia** `{2023:1, 2024:1, 2025:2, 2026:2}`: las filas de temporadas recientes se duplican en el conjunto de entrenamiento. TabNet no acepta `sample_weight` directamente; duplicar filas es equivalente.
- Artefactos en local: `models/tabnet_model.zip` + `models/scaler.pkl` + `models/tabnet_temperature.pkl`

#### Calibración de probabilidades (Temperature Scaling)

Las redes neuronales tienden a ser **sobreconfiadas**: la función softmax final produce probabilidades muy cercanas a 1 aunque la predicción sea incierta. Esto se corrige con **Temperature Scaling**, la técnica de calibración más habitual en producción para redes neuronales ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).

La idea es dividir los logits $z$ (pre-sigmoid) por una temperatura $T \geq 1$ antes de calcular la probabilidad final:

$$p_{\text{cal}} = \sigma\!\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z/T}}$$

- $T = 1$: sin cambio (modelo original)
- $T > 1$: aplana las probabilidades, reduce la sobreconfianza
- $T < 1$: concentra las probabilidades, aumenta la confianza (raro en redes neuronales)

**Cómo se aprende $T$** (`train.py → _fit_temperature`): TabNet se entrena con el **80%** de los datos. Con el **20% restante** (set de calibración) se minimiza la NLL (*Negative Log-Likelihood*) con L-BFGS para encontrar el $T$ óptimo. El resultado se guarda en `models/tabnet_temperature.pkl`.

El mismo mecanismo se aplica a **XGBoost** (`models/xgb_temperature.pkl`) con el mismo esquema 80/20 y la misma función `_fit_temperature`.

**Cómo se aplica**: al predecir, se extraen los logits de las probabilidades brutas ($z = \log\frac{p}{1-p}$), se dividen entre $T$ y se pasa por sigmoid. Si el archivo de temperatura no existe, se usa $T = 1.0$ con un aviso.

> **Razón del diseño dual**: TabNet requiere PyTorch (~1 GB de dependencias), incompatible con los límites prácticos de AWS Lambda. XGBoost es ligero (~50 MB) y se despliega sin problemas. Correr ambos modelos permite comparar sus predicciones directamente.

---

## Infraestructura AWS

| Recurso | Detalles |
|---|---|
| **Región** | `eu-west-1` |
| **S3 bucket** | `f1-winner-predictor-2026` |
| **Lambda** | `f1-winner-predictor` (imagen ECR) |
| **ECR** | `606756239522.dkr.ecr.eu-west-1.amazonaws.com/f1-winner-predictor:latest` |
| **Athena** | DB: `f1_predictions`, tablas: `race_predictions`, `feature_importance` — configurado para consultas SQL ad-hoc; Looker Studio usa Google Sheets como fuente activa |

### Estructura del bucket S3

```
f1-winner-predictor-2026/
├── predictions/
│   └── history.csv              # Una fila por carrera (XGB + TabNet fusionados)
├── models/
│   ├── xgboost_f1_winner.pkl
│   ├── label_encoders.pkl
│   ├── xgb_temperature.pkl      # Temperature Scaling para XGBoost
│   └── tabnet_temperature.pkl   # Temperature Scaling para TabNet
├── metrics/
│   ├── feature_importance.csv   # Importancia de features por modelo
│   └── historical_performance.csv  # Predicciones vs resultados (datos de entrenamiento)
└── data/
    └── race_results_raw.csv     # Datos históricos de entrenamiento
```

### Fusión XGB + TabNet en history.csv

Lambda escribe `predicted_winner_xgb` y el script local escribe `predicted_winner_tab`. La función `append_to_history_csv()` fusiona ambas escrituras en la **misma fila** (indexed by `year` + `round`) sin sobreescribir columnas ya escritas por el otro modelo.

---

## Google Sheets + Looker Studio

Las predicciones y métricas se sincronizan automáticamente con Google Sheets (ID: `1Jw7wo3bqC2IS9MmfSJe6T2waQyp7LTCMv4al7gwhtPI`):

| Pestaña | Contenido | Cuándo se actualiza |
|---|---|---|
| **Sheet1** | `history.csv` — predicciones 2026 por carrera | Al predecir (sábado) y al registrar resultado (lunes) |
| **feature_importance** | Importancia de cada feature en XGBoost y TabNet | Al reentrenar con `--upload-s3` |
| **model_accuracy** | Accuracy, Brier Score y MAE posicional acumulados carrera a carrera | Al registrar el ganador real (lunes) |

### Gráficos en Looker Studio

| Gráfico | Fuente | Configuración |
|---|---|---|
| **Tabla de predicciones** | Sheet1 | Dimensiones: `event_name`, `predicted_winner_xgb`, `predicted_winner_tab`, `actual_winner`, `xgb_correct`, `tab_correct` / Métricas: `win_prob_xgboost`, `win_prob_tabnet` |
| **Barras de importancia** | `feature_importance` | Dimensión Y: `feature` / Métricas X: `importance_xgboost` + `importance_tabnet` |
| **Línea de accuracy** | `model_accuracy` | Dimensión X: `race_label` / Métricas Y: `xgb_accuracy_cumul` + `tab_accuracy_cumul` |
| **Línea de Brier Score** | `model_accuracy` | Dimensión X: `race_label` / Métricas Y: `xgb_brier_cumul` + `tab_brier_cumul` (↓ mejor, mín 0) |
| **Línea de MAE posicional** | `model_accuracy` | Dimensión X: `race_label` / Métricas Y: `xgb_pos_mae_cumul` + `tab_pos_mae_cumul` (↓ mejor, mín 0) |

---

## Métricas de evaluación

Cada vez que se registra el ganador real (`--record-result`), se calculan y sincronizan tres métricas acumuladas para comparar XGBoost y TabNet:

### Accuracy acumulada

Fracción de carreras en las que el modelo acertó el ganador:

$$\text{Accuracy} = \frac{\text{carreras acertadas}}{\text{carreras disputadas}}$$

Columnas: `xgb_accuracy_cumul`, `tab_accuracy_cumul`.

### Brier Score acumulado

Mide la **calibración de probabilidades** — no solo si acertó, sino cuán seguro estaba el modelo cuando acertó o falló. Cuanto más bajo, mejor (0 = perfecto, 1 = completamente equivocado con total confianza):

$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - y_i)^2$$

Donde $p_i$ es la probabilidad asignada al piloto predicho e $y_i \in \{0, 1\}$ indica si realmente ganó.

**Cómo se calcula en el código** (`aws_utils.py → sync_model_accuracy_to_sheets`):
```python
# xgb_brier_i = (win_prob_xgboost - xgb_correct)²
df["xgb_brier"] = (df["win_prob_xgboost"] - df["xgb_correct"]) ** 2
df["xgb_brier_cumul"] = df["xgb_brier"].expanding().mean()
```

Columnas: `xgb_brier`, `xgb_brier_cumul`, `tab_brier`, `tab_brier_cumul`.

### MAE posicional acumulado

Mide **cuántas posiciones se equivocó** el modelo respecto al 1er puesto. Si el modelo predijo a VER y VER terminó P5, el error es 4. Si acertó el ganador, el error es 0:

$$\text{MAE}_{\text{pos}} = \frac{1}{N}\sum_{i=1}^{N}|f_i - 1|$$

Donde $f_i$ es la posición en carrera del piloto que el modelo predijo como ganador.

**Cómo se obtiene** (`predict.py → record_actual_result`): al registrar el resultado real, FastF1 descarga la clasificación de carrera y busca en qué posición terminó el piloto predicho por cada modelo. Ese dato se guarda en `xgb_predicted_finish_pos` / `tab_predicted_finish_pos` dentro de `history.csv`.

```python
# En record_actual_result():
xgb_finish_pos = results[results["Abbreviation"] == xgb_pred]["Position"].values[0]
# Error_i = |pos_final - 1|  →  0 si ganó, 4 si fue P5, etc.
```

Columnas: `xgb_predicted_finish_pos`, `xgb_pos_mae_cumul`, `tab_predicted_finish_pos`, `tab_pos_mae_cumul`.

---

## Configuración inicial (primera vez)

### 1. Variables de entorno

Crea `.env` en la raíz del proyecto (`TFG/`):

```env
AWS_PROFILE=f1-developer
AWS_REGION=eu-west-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### 2. Credenciales de Google Sheets

Descarga el JSON de la service account `f1-sheets-sync@f1-tfg.iam.gserviceaccount.com` y colócalo en:
```
f1-winner-predictor/.google_credentials.json
```
Este archivo está en `.gitignore` y nunca debe subirse al repositorio.

### 3. Construir la imagen Docker

```powershell
cd "TFG"
docker build -t f1-tabnet:latest .
```

---

## Flujo semanal de uso

### Viernes (opcional) — Reentrenar modelos

Solo necesario si hay nuevas carreras disputadas o se quieren mejorar los modelos.

```powershell
# Entrenamiento completo con optimización de hiperparámetros y subida a S3:
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data --optimize

# Sin optimización (más rápido, usa hiperparámetros por defecto):
docker compose run --rm trainer python train.py --model all --upload-s3 --refresh-data

# Solo un modelo:
docker compose run --rm trainer python train.py --model xgboost --upload-s3
docker compose run --rm trainer python train.py --model tabnet
```

Al terminar se actualizan automáticamente:
- XGBoost y encoders en S3 (disponibles para Lambda)
- TabNet y scaler en `models/` local
- Pestaña `feature_importance` de Google Sheets

### Después de la clasificación (sábado) — Predecir

**XGBoost vía Lambda** (invocar manualmente tras la clasificación):

```powershell
aws lambda invoke --function-name f1-winner-predictor `
  --payload '{"year":2026,"round":2}' response.json --region eu-west-1
```

**TabNet vía Docker** (manual, post-clasificación):

```powershell
docker compose run --rm trainer python predict.py --round 2 --year 2026
```

Ambas predicciones se fusionan en una única fila en S3 y se sincronizan con Google Sheets.

### Después de la clasificación (sábado) — Ver probabilidades de todos los pilotos

Para consultar la distribución completa de probabilidades de victoria de todos los pilotos **sin escribir nada en S3**, usa el servicio `proba`:

```powershell
docker compose run --rm proba --round 3
docker compose run --rm proba --round 3 --year 2026
```

El script carga los modelos locales (`models/`) y muestra una tabla ordenada por probabilidad XGBoost. La tabla se guarda automáticamente en `data/processed/proba_table_{year}_R{round:02d}.csv`:

```
+==================================================+
|      2026  |  Ronda 3  |  JAPANESE GRAND PRIX    |
+==================================================+
  #   PILOTO      XGBoost     TabNet
  ----------------------------------------------------
  1   ANT           60.0%      50.0% <--
  2   RUS           21.1%      47.6%
  3   PIA            4.2%       0.7%
  4   HAM            2.3%       0.2%
  5   VER            2.1%       0.0%
  ...
[INFO] Tabla guardada en: data/processed/proba_table_2026_R03.csv
```

> Las probabilidades de cada modelo suman 100% tras normalizar las salidas del clasificador binario. TabNet distribuye más probabilidad entre varios pilotos mientras que XGBoost tiende a concentrarla más en el favorito.

### Después de la carrera (lunes) — Registrar resultado real

```powershell
# FastF1 descarga automáticamente el ganador desde la API:
docker compose run --rm trainer python predict.py --round 2 --year 2026 --record-result
```

Esto:
1. Descarga el resultado desde FastF1
2. Escribe `actual_winner`, `xgb_correct`, `tab_correct`, `xgb_predicted_finish_pos` y `tab_predicted_finish_pos` en S3
3. Sincroniza Sheet1 con el resultado real
4. Actualiza la pestaña `model_accuracy` con accuracy, Brier Score y MAE posicional acumulados
5. Guarda `metrics/model_accuracy/model_accuracy.csv` en S3 para Athena

---

## Despliegue (`deploy.ps1`)

El script `deploy.ps1` automatiza todo el pipeline de despliegue. Acepta cuatro modos:

| Modo | Qué hace |
|---|---|
| `models` | Reentrena XGBoost + TabNet y sube artefactos a S3 (sin tocar Lambda) |
| `lambda` | Rebuild imagen Docker → push ECR → update función Lambda (sin reentrenar) |
| `predict` | Invoca Lambda (XGBoost) + ejecuta TabNet local, genera snapshot y CSV de probabilidades |
| `all` | `models` + `lambda` + `predict` en orden |

```powershell
# Solo redesplegar Lambda (código cambiado, modelo no):
.\deploy.ps1 -Mode lambda

# Reentrenar y subir modelos a S3 (sin rebuild Lambda):
.\deploy.ps1 -Mode models

# Pipeline completo con datos frescos:
.\deploy.ps1 -Mode all -Refresh

# Pipeline completo con optimización de hiperparámetros:
.\deploy.ps1 -Mode all -Refresh -Optimize
```

> Los archivos `.py` se montan como volúmenes durante el entrenamiento, por lo que los cambios locales son inmediatos sin necesidad de reconstruir la imagen Docker del trainer.

---

## Fins de semana sprint

En un sprint weekend el calendario difiere del habitual:

| Sesión | GP normal | GP sprint |
|---|---|---|
| Viernes mañana | FP1 | FP1 |
| Viernes tarde | FP2 | Sprint Qualifying (SQ) |
| Sábado mañana | FP3 | Sprint Race (S) |
| Sábado tarde | Qualifying (Q) | Qualifying (Q) |

Esto genera dos problemas estructurales para el modelo si no se trata explícitamente:

### 1. FP2/FP3 no existen → fallback a FP1

`fetch_practice_pace` intenta cargar FP2 y FP3. En un sprint weekend ambas sesiones no existen y FastF1 lanza una excepción. Sin corrección, `fp2_long_run_pace_gap_s` y `fp3_gap_to_best_s` se rellenarían con `0` para **todos** los pilotos tras el `fillna(0)` de `apply_features`, borrando toda la información de ritmo de práctica y haciendo que el modelo no pueda diferenciar a los pilotos por pace.

**Fix implementado** (`fetch_practice_pace`): cuando ambas sesiones vuelven vacías, se carga FP1. Los gaps de mejor vuelta de FP1 por piloto (respecto al más rápido) se usan como sustituto para ambas columnas. Es un proxy degradado pero real —FP1 tiene datos de ritmo real, al contrario que un cero artificial.

### 2. Puntos del Sprint Race no contabilizados en el campeonato

`race_results_raw.csv` solo almacenaba puntos de la carrera principal. En un sprint weekend los pilotos obtienen puntos adicionales del Sprint Race (hasta 8 pts por P1 desde 2023). Si esos puntos no se incluyen, las features `driver_champ_points` y `constructor_champ_points` quedan subestimadas a partir de esa ronda, contaminando el cálculo de standings de todas las carreras siguientes de la temporada.

**Fix implementado** (`fetch_season_results`): se intenta cargar la sesión `"S"` (Sprint Race) para cada ronda. Si existe, los puntos de sprint de cada piloto se suman a los de la carrera principal en la columna `points` de `race_results_raw.csv`. Como `_championship_standings` y `_enrich_live_from_history` suman esa columna directamente, los standings quedan correctos de forma automática sin cambios en `feature_engineering.py`.

### Impacto del fix

| Feature | Sin fix (sprint weekend) | Con fix |
|---|---|---|
| `fp2_long_run_pace_gap_s` | `0` para todos | Gap real desde FP1 |
| `fp3_gap_to_best_s` | `0` para todos | Gap real desde FP1 |
| `sprint_race_position` | `0` (sin sprint) | Posición real del Sprint Race |
| `driver_champ_points` | Subestimado | Exacto (main race + sprint) |
| `constructor_champ_points` | Subestimado | Exacto (main race + sprint) |

> **Nota**: Tras cualquier cambio en `data_collection.py` hay que regenerar `race_results_raw.csv` con `--refresh-data` para que los datos históricos reflejen el fix.

---

## Seguridad

- Las credenciales AWS van en `.env` (gitignored) o en variables de entorno del sistema
- Las credenciales de Google van en `.google_credentials.json` (gitignored)
- Los artefactos de modelo (`models/*.pkl`, `models/*.zip`) y los datos (`data/`) están en `.gitignore`
- El directorio `package/` (build de Lambda) está en `.gitignore`
- Nunca se hardcodean claves en el código fuente

