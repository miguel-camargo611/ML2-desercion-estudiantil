# Segunda entrega — Fase 2: datos, modelamiento y validación

| Archivo | Contenido |
|---|---|
| `segunda_entrega.pdf` | Informe compilado (portada, 4 secciones, referencias) |
| `segunda_entrega.tex` | Fuente LaTeX del informe |
| `tabla_resultados.tex` | Tabla de resultados de la CV anidada, **generada** por `desercion.cli report` |
| `figuras/` | Figuras del informe, **generadas** por `desercion.cli figures` y copiadas por `report` |

## Regenerar

Desde la raíz del repositorio:

```bash
python -m desercion.cli all      # datos -> modelos -> experimentos -> SHAP -> figuras -> tablas
```

Para compilar el informe basta subir a Overleaf `segunda_entrega.tex`, `tabla_resultados.tex`
y la carpeta `figuras/` (compilador pdfLaTeX o XeLaTeX), o usar `tectonic segunda_entrega.tex`.

## Correspondencia con la guía (sección 7)

| Guía | Informe | Código |
|---|---|---|
| 7.1 Consolidación de los datos | Sección 1, Figuras 1–2, Tabla 1 | `data.py`, `features.py` |
| 7.2 Flujo sin fuga | Sección 2, Figura 3 | `models.py` (`build_pipeline`), `evaluate.py`, `experiments.leakage_demo` |
| 7.3 Reproducción y adaptación | Sección 3, Tablas 2–3, Figuras 4–5 | `models.py`, `experiments.py` |
| 7.4 Validación, métricas e interpretación | Sección 4, Figuras 6–8 | `evaluate.py`, `interpret.py` |
