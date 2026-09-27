"""Carga, limpieza, definición del objetivo, partición y escenarios de modelamiento.

Fuente: Alvarado-Uribe et al. (2022), *Student Dataset from Tecnologico de Monterrey in
Mexico to Predict Dropout in Higher Education*, TEC DataHub (doi:10.57687/FK2/PWJRSJ).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from desercion import config

log = logging.getLogger(__name__)

TARGET = "desercion"                 # 1 = deserta en el primer año, 0 = retenido
TARGET_MULTI = "desercion_momento"   # 0 retenido, 1 deserción 1er semestre, 2 deserción 2º semestre
GROUP = "student.id"
MULTI_LABELS = {0: "Retenido", 1: "Deserción 1er sem.", 2: "Deserción 2º sem."}

# Códigos textuales que el diccionario usa para ausencias.
NO_INFO = "No information"
NOT_APPLY = "Does not apply"


# --------------------------------------------------------------------------- carga
def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Lee el Excel original y lo guarda en parquet (la lectura del .xlsx tarda ~80 s)."""
    cache = config.path("interim_data")
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    raw_path = config.path("raw_data")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No se encontró {raw_path}. Descargue el dataset del TEC DataHub y ubíquelo en data/raw/ "
            "(ver data/README.md)."
        )
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
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    for col in ["admission.test", "general.math.eval"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df["english.evaluation"] = df["english.evaluation"].replace(0, np.nan).astype(float)
    df["PNA"] = df["PNA"].replace(0, np.nan)

    for col in df.select_dtypes(include=["string"]).columns:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Crea el objetivo binario y la versión de tres clases (análoga a la del artículo)."""
    df = df.copy()
    df[TARGET] = (1 - df["retention"]).astype(int)
    df[TARGET_MULTI] = df["dropout.semester"].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2}).astype(int)
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


# ----------------------------------------------------------------------- escenarios
SCENARIOS = {
    "A": "Al ingreso (AD14–AD20, solo información de admisión)",
    "B": "Fin del primer periodo (AD19–AD20, admisión + desempeño del primer periodo)",
}


def scenario_frame(df: pd.DataFrame, scenario: str, split: pd.Series, part: str = "dev") -> pd.DataFrame:
    """Filtra filas de un escenario y una parte (``dev`` o ``test``).

    En el escenario B se excluye a quien desertó *durante* el primer semestre
    (``dropout.semester == 1``): en el momento de la predicción ya no está matriculado y su
    promedio del periodo refleja el abandono, no lo anticipa (fuga temporal).
    """
    out = df[split.reindex(df.index) == part]
    if scenario == "B":
        gens = config.load_config()["first_period_generations"]
        out = out[out["generation"].isin(gens) & (out["dropout.semester"] != 1)]
    elif scenario != "A":
        raise ValueError(f"Escenario desconocido: {scenario}")
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
