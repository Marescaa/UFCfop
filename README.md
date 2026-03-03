# 🥊 UFC Fight Outcome Predictor

Modelo de clasificación binaria que predice si el peleador de la esquina roja gana un combate de UFC, usando únicamente estadísticas históricas previas a la pelea.

---

## Problema

Dado el historial estadístico de dos peleadores antes de un combate, ¿es posible predecir el resultado mejor que el azar?

La dificultad principal no es técnica sino metodológica: la mayoría de los datasets de UFC contienen estadísticas *de la pelea misma* (golpes conectados, takedowns en ese combate), que no estarían disponibles en el momento de predecir. Usarlas genera **data leakage** y produce modelos con accuracy artificialmente alta que no generalizan.

Este proyecto trata ese problema de forma explícita.

---

## Dataset

- **Fuente:** Dataset público de peleas históricas de UFC (~8300 combates)
- **Archivos:** `UFC.csv`, `fight_details.csv`, `event_details.csv`, `fighter_details.csv`
- **Target:** `red_win` — variable binaria (1 = gana esquina roja, 0 = gana esquina azul)
- **Distribución:** 55.4% Red wins / 44.6% Blue wins

---

## Decisiones metodológicas

### Features utilizadas

Se usó una **whitelist explícita** de 16 estadísticas históricas pre-fight, 8 por peleador:

| Feature | Descripción |
|---|---|
| `str_acc` | Striking accuracy histórica |
| `sapm` | Strikes absorbidos por minuto |
| `splm` | Strikes lanzados por minuto |
| `str_def` | Defensa de strikes |
| `td_avg` | Promedio de takedowns |
| `td_avg_acc` | Accuracy de takedowns |
| `td_def` | Defensa de takedowns |
| `sub_avg` | Promedio de intentos de sumisión |

Se descartó el filtrado por keywords porque es frágil ante cambios en el dataset. Cualquier feature nueva debe agregarse conscientemente a la whitelist.

### Feature engineering

Para cada par `r_stat` / `b_stat` se construyó la diferencia directa `stat_diff = r_stat - b_stat`. Los modelos lineales capturan diferencias entre oponentes de forma más eficiente que a partir de valores individuales.

### Validación temporal

Se ordenaron los datos cronológicamente y se usaron dos estrategias de validación:

- **TimeSeriesSplit (5 folds):** para selección de modelo. Entrena siempre con el pasado y evalúa en el futuro inmediato.
- **Split temporal 80/20:** para métricas finales y visualizaciones. Train: 6669 peleas, Test: 1668 peleas.

No se usó `train_test_split` aleatorio porque mezclaría peleas futuras en el entrenamiento, lo que constituye leakage temporal.

---

## Resultados

### Comparación de modelos (TimeSeriesCV)

| Modelo | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | **0.667** | 0.030 |
| Random Forest | 0.667 | 0.027 |
| XGBoost | 0.650 | 0.020 |
| Baseline (mayoría) | 0.500 | — |

### Evaluación final (split temporal 80/20)

| Métrica | Valor |
|---|---|
| Baseline accuracy | 0.554 |
| **Accuracy** | **0.651** |
| **AUC-ROC** | **0.712** |

El modelo supera al baseline en **+19.7 puntos de AUC** usando únicamente estadísticas históricas disponibles antes del combate.

> **Nota sobre el AUC:** Una versión previa del proyecto reportaba AUC ~0.82 usando estadísticas de la pelea misma (golpes conectados, takedowns del combate). Esos valores son inflados por leakage y no representan capacidad predictiva real. El AUC 0.712 reportado aquí es honesto.

---

## Estructura del proyecto

```
ufc-fight-predictor/
│
├── UFC_predictor.ipynb     ← notebook principal
├── data/
│   ├── UFC.csv
│   ├── fight_details.csv
│   ├── event_details.csv
│   └── fighter_details.csv
├── results/
│   ├── evaluation_plots.png
│   ├── calibration_curve.png
│   └── feature_importance.png
└── README.md
```

---

## Cómo reproducir

```bash
# 1. Clonar el repositorio
git clone https://github.com/Marescaa/ufc-fight-predictor.git
cd ufc-fight-predictor

# 2. Instalar dependencias
pip install pandas numpy matplotlib scikit-learn xgboost

# 3. Colocar los archivos CSV en data/

# 4. Abrir y correr el notebook
jupyter notebook UFC_predictor.ipynb
```

---

## Limitaciones y trabajo futuro

El modelo actual usa promedios históricos totales. Las mejoras con mayor impacto potencial serían:

- **Forma reciente:** promedio de las últimas 3-5 peleas en vez del total histórico
- **Experiencia:** cantidad de peleas previas como proxy de rodaje
- **Tiempo inactivo:** días desde la última pelea antes del combate
- **Odds del mercado:** las probabilidades implícitas de apuestas son el benchmark real en predicción deportiva

---

## Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib`
