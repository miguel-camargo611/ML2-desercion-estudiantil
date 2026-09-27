# Predicción de deserción estudiantil — Tec de Monterrey

Proyecto integrador de **Machine Learning II** (Universidad Externado de Colombia, 2026-2).
Reproduce y adapta el método de Villar & Andrade (2024), *Supervised machine learning algorithms
for predicting student dropout and academic success: a comparative study*, sobre datos reales
latinoamericanos: el *Student Dropout Dataset* del Tecnológico de Monterrey (Alvarado-Uribe et al., 2022).

**Integrantes:** Miguel Camargo · Camilo Hernández  **Profesor:** Wilmer Pineda

| Entrega | Fase | Carpeta |
|---|---|---|
| Primera | Comprensión del artículo y planteamiento | [`Primera_entrega/`](Primera_entrega/) |
| Segunda | Datos, modelamiento y validación | [`Segunda_entrega/`](Segunda_entrega/) |
| Final | Consolidación, despliegue (Streamlit) y comunicación | *pendiente* |

## Pregunta y diseño

¿Qué tan bien se anticipa la deserción en el primer año de pregrado y cuánto aporta el contexto
socioeconómico frente al desempeño académico temprano? Clasificación binaria (`desercion = 1 - retention`,
8,8 % de positivos) en dos momentos de predicción:

- **Escenario A — al ingreso** (cohortes AD14–AD20): solo información de admisión, como Martins et al. (2021).
- **Escenario B — fin del primer periodo** (AD19–AD20): admisión + desempeño del primer periodo, como Villar & Andrade (2024).

Se comparan árbol sin poda y clasificador de mayoría (líneas base), regresión logística, Random Forest,
SVM con kernel RBF (aproximación de Nystroem), Gradient Boosting, XGBoost, LightGBM y CatBoost
(con SMOTE o con categóricas nativas), con **validación cruzada anidada** (5×3) agrupada por estudiante,
SMOTE dentro del `Pipeline` y umbral de decisión calibrado para priorizar el *recall* (F2).

## Estructura

```
├── configs/config.yaml        # semilla, rutas, esquema de validación, métricas
├── data/                      # raw/ interim/ processed/ (no versionados, ver data/README.md)
├── notebooks/                 # exploración (la lógica vive en src/)
├── reports/
│   ├── figures/               # figuras generadas por el flujo
│   └── tables/                # resultados en CSV (métricas por pliegue, pruebas, SHAP…)
├── src/desercion/
│   ├── data.py                # carga, limpieza, objetivo, partición sellada, escenarios
│   ├── features.py            # variables por escenario, exclusiones y preprocesamiento
│   ├── models.py              # catálogo de modelos, rejillas y Pipeline con SMOTE
│   ├── evaluate.py            # CV anidada, umbral F2, métricas y pruebas estadísticas
│   ├── experiments.py         # SMOTE vs ADASYN, fuga, Isolation Forest, ablación, multiclase, VIF
│   ├── interpret.py           # SHAP por pliegue y análisis de errores
│   ├── plots.py               # figuras del informe
│   └── cli.py                 # punto de entrada del flujo
├── tests/                     # pruebas con datos sintéticos (corren en CI)
├── Primera_entrega/  Segunda_entrega/   # informes (LaTeX)
└── .github/workflows/ci.yml   # lint + pruebas en cada push
```

## Cómo reproducir

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
# Ubicar el Excel del TEC DataHub en data/raw/ (ver data/README.md)
python -m desercion.cli all       # o paso a paso: prepare, train, compare, experiments, interpret, figures, report
pytest                            # pruebas
```

La corrida completa tarda unas 4 horas en un portátil de 12 hilos (la CV anidada del escenario A es
lo más costoso). Todo usa la semilla `42` de `configs/config.yaml`. El 20 % de los estudiantes queda
en un **conjunto de prueba sellado** que solo se usará en la Entrega final.

## Buenas prácticas adoptadas

- Datos crudos inmutables y fuera de git; transformaciones en código, nunca a mano.
- Configuración centralizada y semilla fija; dependencias declaradas en `pyproject.toml`.
- Todo el preprocesamiento dentro de `Pipeline` (sin fuga), validado con pruebas automáticas.
- Resultados como artefactos regenerables (`reports/`), con el comando que los produce.
- Integración continua (ruff + pytest) y ramas/PR para cambios.

## Declaración de uso de IA

Se usó un asistente de IA generativa (Claude, Anthropic) como apoyo en la escritura de código y
la revisión del texto. El equipo revisó, ejecutó y puede sustentar cada decisión técnica.

## Referencias

- Villar, A. & Robledo Velini de Andrade, C. (2024). *Discover Artificial Intelligence*, 4:2. https://doi.org/10.1007/s44163-023-00079-z
- Alvarado-Uribe, J. et al. (2022). *Data*, 7(9), 119. https://doi.org/10.3390/data7090119
- Martins, M. V. et al. (2021). *Trends and Applications in Information Systems and Technologies*, vol. 9, 166–175.

Código bajo licencia MIT (ver `LICENSE`). Los datos pertenecen al Tecnológico de Monterrey.
