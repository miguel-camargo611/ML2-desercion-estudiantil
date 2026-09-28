# Segunda entrega — Fase 2: datos, modelamiento y validación

| Archivo | Contenido |
|---|---|
| `segunda_entrega.pdf` | Informe compilado |
| `segunda_entrega.tex` | Fuente LaTeX del informe |
| `tabla_*.tex` | Tablas de resultados, hiperparámetros y fuga, **generadas** por `desercion.cli report` |
| `figuras/` | Figuras del informe, **generadas** por el flujo y copiadas por `report` |

## Regenerar

Desde la raíz del repositorio:

```bash
python -m desercion.cli all      # datos -> modelos -> experimentos -> SHAP -> figuras -> tablas
```

Para compilar el informe basta subir a Overleaf `segunda_entrega.tex`, los `tabla_*.tex`
y la carpeta `figuras/` (compilador pdfLaTeX o XeLaTeX), o usar `tectonic segunda_entrega.tex`.

## Correspondencia con la guía (sección 7)

| Guía | Informe | Código |
|---|---|---|
| 7.1 Consolidación de los datos | Sección 1, Figura 1, Tablas 1–2 | `data.py`, `features.py` |
| 7.2 Flujo sin fuga | Sección 2, Tabla 3 | `models.py` (`build_pipeline`), `evaluate.py`, `experiments.leakage_demo` |
| 7.3 Reproducción y adaptación | Sección 3, Tablas 4–6 | `models.py`, `evaluate.py`, `experiments.py` |
| 7.4 Validación, métricas e interpretación | Sección 4, Figuras 2–3 | `evaluate.py`, `experiments.ablation`, `interpret.py` |
