# Datos

Los datos **no se versionan** en este repositorio: su uso está sujeto a los términos del
TEC DataHub (Tecnológico de Monterrey), que los entrega previa solicitud.

| Carpeta | Contenido | ¿Quién la escribe? |
|---|---|---|
| `raw/` | Archivo original, inmutable: `2022-03-16 dataset on student dropout .xlsx` | Se descarga a mano |
| `interim/` | `students.parquet`: copia del Excel en formato rápido (sin transformar) | `desercion prepare` |
| `processed/` | Partición desarrollo/prueba (`split.parquet`), predicciones fuera de pliegue y SHAP | pasos del flujo |

## Cómo obtener el dataset

1. Solicitarlo en el TEC DataHub: *Student Dropout Dataset*, DOI
   [10.57687/FK2/PWJRSJ](https://doi.org/10.57687/FK2/PWJRSJ).
2. Guardar el Excel recibido, sin renombrarlo, en `data/raw/`.
3. Ejecutar `python -m desercion.cli prepare`.

## Descripción

Alvarado-Uribe et al. (2022), *Data* 7(9):119. 143.326 registros (121.584 estudiantes),
cohortes AD14–AD20, bachillerato y pregrado, 50 variables. El proyecto usa solo **pregrado**
(77.517 registros). El diccionario de variables está en el archivo *teaser* público del DataHub.

## Conjunto de prueba sellado

`split.parquet` separa un 20 % de los estudiantes (estratificado por cohorte y objetivo,
agrupado por `student.id`) que **no se usa** hasta la Entrega final.
