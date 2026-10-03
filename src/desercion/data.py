"""Carga, limpieza, definición del objetivo, partición y población de estudio.

Fuente: Alvarado-Uribe et al. (2022), *Student Dataset from Tecnologico de Monterrey in
Mexico to Predict Dropout in Higher Education*, TEC DataHub (doi:10.57687/FK2/PWJRSJ).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from desercion import config

log = logging.getLogger(__name__)

TARGET = "desercion"                 # 1 = deserta en el primer año, 0 = retenido
GROUP = "student.id"

# Códigos textuales que el diccionario usa para ausencias.
NO_INFO = "No information"
NOT_APPLY = "Does not apply"


# --------------------------------------------------------------------------- carga
def find_raw_file() -> Path:
    """Ubica el Excel original en ``data/raw/`` (basta con que haya un único archivo .xlsx)."""
    raw_dir = config.path("raw_dir")
    files = sorted(raw_dir.glob("*.xlsx"))
    if len(files) == 1:
        return files[0]
    found = "ningún archivo .xlsx" if not files else f"{len(files)} archivos .xlsx"
    raise FileNotFoundError(
        f"Se encontró {found} en {raw_dir}. Solicite el Student Dropout Dataset al TEC DataHub "
        "(https://doi.org/10.57687/FK2/PWJRSJ) y copie el Excel recibido, con cualquier nombre, "
        "en data/raw/. Debe haber un solo archivo .xlsx en esa carpeta."
    )


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Lee el Excel original y lo guarda en parquet (la lectura del .xlsx tarda ~80 s)."""
    cache = config.path("interim_data")
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    raw_path = find_raw_file()
    log.info("Leyendo %s", raw_path)
    df = pd.read_excel(raw_path)
    df.index.name = "row_id"
    df = df.astype({c: "string" for c in df.select_dtypes(include=["object", "string"]).columns})
    df.to_parquet(cache)
    return df


# ------------------------------------------------------------------------ limpieza
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos y ausencias sin usar información del objetivo (transformación fila a fila).

    - Recorta espacios (p. ej. ``'Does not apply '``).
    - Convierte a numéricas las variables mixtas número/texto; el texto de ausencia pasa a NaN.
    - ``english.evaluation == 0`` y ``PNA == 0`` significan "sin información" -> NaN.
    - El examen de admisión tiene dos escalas: el presencial va de 400 a 1600 y el examen en
      línea (``online.test == 1``) de 0 a 100. Mezclarlas en una sola columna haría que un
      puntaje en línea pareciera extremadamente bajo, así que se separan en dos variables.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    for col in ["admission.test", "general.math.eval"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    online = df["admission.test"] <= 100
    df["admission.test.online"] = df["admission.test"].where(online)
    df["admission.test"] = df["admission.test"].mask(online)

    df["english.evaluation"] = df["english.evaluation"].replace(0, np.nan).astype(float)
    df["PNA"] = df["PNA"].replace(0, np.nan)

    for col in df.select_dtypes(include=["string"]).columns:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Crea el objetivo binario: 1 si el estudiante ya no está matriculado al terminar el primer año."""
    df = df.copy()
    df[TARGET] = (1 - df["retention"]).astype(int)
    return df


def load_clean() -> pd.DataFrame:
    """Datos limpios de la población de estudio (pregrado) con los objetivos definidos."""
    cfg = config.load_config()
    df = add_targets(clean(load_raw()))
    return df[df["level"] == cfg["population"]["level"]].copy()


# ------------------------------------------------------------------------ partición
def make_split(df: pd.DataFrame, test_size: float | None = None, seed: int | None = None) -> pd.Series:
    """Partición desarrollo/prueba estratificada (objetivo x cohorte) y agrupada por estudiante.

    Agrupar por ``student.id`` garantiza que un mismo estudiante nunca quede a ambos lados.
    """
    cfg = config.load_config()
    test_size = cfg["split"]["test_size"] if test_size is None else test_size
    seed = config.seed() if seed is None else seed
    n_splits = int(round(1 / test_size))
    strat = df["generation"].astype(str) + "_" + df[TARGET].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, strat, groups=df[GROUP]))
    split = pd.Series("dev", index=df.index, name="split")
    split.iloc[test_idx] = "test"
    return split


def get_split(df: pd.DataFrame, recompute: bool = False) -> pd.Series:
    p = config.path("split")
    if p.exists() and not recompute:
        split = pd.read_parquet(p)["split"]
        return split.reindex(df.index)
    split = make_split(df)
    split.to_frame().to_parquet(p)
    return split


# ------------------------------------------------------------- población de estudio
def study_frame(df: pd.DataFrame, split: pd.Series, part: str = "dev") -> pd.DataFrame:
    """Estudiantes de la población de estudio en una parte de la partición (``dev`` o ``test``).

    La predicción se hace al terminar el primer semestre, con la información de admisión y el
    desempeño en ese semestre, como en Villar y Andrade (2024). Por eso:

    - Solo entran las cohortes que registran el desempeño del primer semestre (AD19 y AD20).
    - Se excluye a quien desertó *durante* el primer semestre (``dropout.semester == 1``): en el
      momento de la predicción ya no está matriculado y su promedio refleja el abandono, no lo
      anticipa (fuga temporal).
    """
    gens = config.load_config()["study_generations"]
    out = df[(split.reindex(df.index) == part) & df["generation"].isin(gens)
             & (df["dropout.semester"] != 1)]
    return out.copy()


# ------------------------------------------------------------------ calidad de datos
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de calidad por variable: tipo, % NaN, % 'No information', % 'Does not apply'."""
    rows = []
    for col in df.columns:
        s = df[col]
        as_str = s.astype(str)
        rows.append(
            {
                "variable": col,
                "tipo": "numérica" if pd.api.types.is_numeric_dtype(s) else "categórica",
                "n_unicos": s.nunique(dropna=True),
                "pct_nan": 100 * s.isna().mean(),
                "pct_no_information": 100 * (as_str == NO_INFO).mean(),
                "pct_does_not_apply": 100 * (as_str == NOT_APPLY).mean(),
            }
        )
    return pd.DataFrame(rows).round(2)
