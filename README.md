# Predicción de la deserción estudiantil — Tec de Monterrey

Proyecto integrador de **Machine Learning II** (Universidad Externado de Colombia, 2026-2).
Reproduce y adapta el método de Villar y Andrade (2024), *Supervised machine learning algorithms
for predicting student dropout and academic success: a comparative study*, sobre datos reales
latinoamericanos: el *Student Dropout Dataset* del Tecnológico de Monterrey (Alvarado-Uribe et al., 2022).

**Integrantes:** Miguel Camargo · Camilo Hernández  **Profesor:** Wilmer Pineda

| Entrega | Fase | Informe |
|---|---|---|
| Primera | Comprensión del artículo y planteamiento | [`reports/primera_entrega/`](reports/primera_entrega/primera_entrega.pdf) |
| Segunda | Datos, modelamiento y validación | [`reports/segunda_entrega/`](reports/segunda_entrega/segunda_entrega.pdf) |
| Final | Consolidación, despliegue (Streamlit) y comunicación | *pendiente* |

## Problema

Al terminar el primer semestre, con la información de admisión y el desempeño en ese semestre,
predecir qué estudiantes de pregrado ya no estarán matriculados al cerrar su primer año, y
determinar qué modelo lo hace mejor. Clasificación binaria con 7,8 % de positivos, sobre las
cohortes AD19–AD20 (24.251 estudiantes, las únicas que registran el desempeño del primer semestre).

## Resultados (Segunda entrega)

Validación cruzada anidada (5 × 3) en el conjunto de desarrollo; el 20 % de prueba sigue sellado.

| Modelo | PR-AUC (media ± DE) | ROC-AUC | Ajuste del modelo final |
|---|---|---|---|
| **CatBoost** | **0,248 ± 0,023** | 0,711 | 16,0 s |
| XGBoost | 0,248 ± 0,021 | 0,717 | 1,0 s |
| LightGBM | 0,247 ± 0,023 | 0,716 | 0,5 s |
| Gradient Boosting | 0,245 ± 0,022 | 0,716 | 1,2 s |
| SVM (kernel RBF) | 0,235 ± 0,017 | 0,716 | 2,4 s |
| Random Forest | 0,230 ± 0,018 | 0,697 | 4,6 s |
| Regresión logística | 0,224 ± 0,019 | 0,715 | 0,3 s |
| Árbol sin poda (línea base) | 0,091 ± 0,005 | 0,546 | 0,4 s |
| Clasificador de mayoría (línea base) | 0,078 | 0,500 | 0,2 s |

El PR-AUC de un modelo al azar es 0,078 (la proporción de desertores). Alertando al 20 % de
estudiantes con mayor riesgo, CatBoost detecta al 46 % de los desertores.

- **La familia de *boosting* empata** (prueba t corregida de Nadeau-Bengio, *p* > 0,4) y CatBoost
  supera por poco, pero de forma significativa, a la regresión logística (*p* = 0,026). La señal es débil
  (ROC-AUC ≈ 0,72). Con desempeños empatados, el costo computacional sirve de desempate.
- **SMOTE no ayuda:** al tratarlo como hiperparámetro, la validación lo rechazó en las 30 elecciones
  (6 modelos × 5 rondas).
- **Fuga en el artículo:** aplicar SMOTE antes de separar los datos, como hacen Villar y Andrade,
  sube el F1 de la clase deserción de 0,12 a 0,95, lo que sugiere que parte del desempeño publicado
  proviene del protocolo y no del modelo.
- **El promedio del primer semestre es el mejor predictor**, como en el artículo.

## Cómo reproducir

**1. Entorno** (Python ≥ 3.11; las versiones de las dependencias están fijadas en `pyproject.toml`):

```bash
git clone https://github.com/miguel-camargo611/ML2-desercion-estudiantil.git
cd ML2-desercion-estudiantil
python -m venv .venv
.venv\Scripts\activate          # Windows   ·   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

**2. Datos.** Los datos no se incluyen: su uso está sujeto a los términos del TEC DataHub.
Solicite el *Student Dropout Dataset* ([doi:10.57687/FK2/PWJRSJ](https://doi.org/10.57687/FK2/PWJRSJ))
y copie el Excel recibido, **con cualquier nombre**, en `data/raw/`. Debe ser el único `.xlsx` de
esa carpeta; si falta, el programa lo indica.

**3. Flujo completo** (≈ 2 horas en un portátil de 12 hilos):

```bash
desercion all
```

o paso a paso:

| Paso | Qué hace | Salidas |
|---|---|---|
| `desercion prepare` | Limpieza, población de estudio y partición desarrollo/prueba | `data/processed/`, `reports/tables/calidad_datos.csv` |
| `desercion eda` | Análisis exploratorio del conjunto de desarrollo | `reports/{tables,figures}/eda/` |
| `desercion select` | Variables construidas y selección de variables | `reports/tables/seleccion_*.csv` |
| `desercion train` | Validación cruzada anidada de los 9 modelos | `reports/tables/cv_*.csv` |
| `desercion compare` | Pruebas estadísticas entre modelos | `pruebas_estadisticas.csv` |
| `desercion experiments` | Réplicas del artículo, curvas de validación y costo computacional | `fuga_smote.csv`, `costo_computacional.csv`… |
| `desercion interpret` | SHAP por pliegue, errores y captura por nivel de riesgo | `shap.csv`, `errores.csv`… |
| `desercion figures` / `report` | Figuras y tablas LaTeX del informe | `reports/figures/`, `reports/tables/*.tex` |

Para una prueba rápida: `desercion train --sample 0.1`. Las pruebas automáticas (`pytest`) usan
datos sintéticos y no requieren el dataset; corren en cada *push* (GitHub Actions).

**Semilla.** Todo usa la semilla `42` de `configs/config.yaml`: la partición, la validación cruzada,
SMOTE y cada modelo reciben ese `random_state`, y la CLI fija además las semillas de Python y NumPy.

## Estructura

```
├── configs/config.yaml        # semilla, rutas, validación y métricas
├── data/raw/                  # aquí va el Excel del TEC (no se versiona)
├── notebooks/                 # EDA comentado
├── reports/
│   ├── figures/  tables/      # resultados generados por el flujo (CSV, PNG, tablas LaTeX)
│   └── primera_entrega/  segunda_entrega/   # informes (LaTeX y PDF)
├── src/desercion/
│   ├── data.py                # carga, limpieza, objetivo, partición y población de estudio
│   ├── eda.py                 # análisis exploratorio
│   ├── features.py            # variables, exclusiones, variables construidas y preprocesamiento
│   ├── models.py              # modelos, rejillas y Pipeline (preprocesamiento → balanceo → modelo)
│   ├── evaluate.py            # CV anidada, umbral F2, métricas y pruebas estadísticas
│   ├── experiments.py         # selección, réplicas del artículo, curvas de validación y costo
│   ├── interpret.py           # SHAP y análisis de errores
│   ├── plots.py · report.py   # figuras y tablas del informe
│   └── cli.py                 # punto de entrada (`desercion <paso>`)
└── tests/                     # pruebas con datos sintéticos
```

## Uso de IA generativa

Se usó un asistente de IA (Claude, de Anthropic) como apoyo en la escritura de código y la
revisión del texto. El equipo revisó, ejecutó y puede sustentar cada decisión técnica.

## Referencias

- Villar, A. y Robledo Velini de Andrade, C. (2024). *Discover Artificial Intelligence*, 4:2. https://doi.org/10.1007/s44163-023-00079-z
- Alvarado-Uribe, J. et al. (2022). *Data*, 7(9), 119. https://doi.org/10.3390/data7090119

Código bajo licencia MIT. Los datos pertenecen al Tecnológico de Monterrey.
