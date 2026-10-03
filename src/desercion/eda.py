"""Análisis exploratorio (EDA) del conjunto de desarrollo.

Se hace solo sobre la parte de desarrollo: el conjunto de prueba sellado no se mira ni
siquiera de forma descriptiva, para que ninguna decisión de modelamiento dependa de él.

Todas las salidas son agregados (tasas, conteos, cuantiles); nunca registros individuales,
porque los datos del TEC no pueden redistribuirse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from desercion.data import NO_INFO, TARGET

# Etiquetas del momento de la deserción según el diccionario del dataset.
MOMENTOS = {
    1: "Durante el 1.er semestre",
    2: "No se inscribe al 2.º semestre",
    3: "Durante el 2.º semestre",
    4: "No se inscribe al 3.er semestre",
}


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de confianza de Wilson para una proporción (estable con tasas bajas)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return np.clip(centro - margen, 0, 1), np.clip(centro + margen, 0, 1)


def is_missing(s: pd.Series) -> pd.Series:
    """Ausente = NaN o el código textual 'No information'."""
    return s.isna() | (s.astype(str) == NO_INFO)


# ------------------------------------------------------------------ objetivo
def target_by_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de deserción por cohorte, desagregada por el momento en que ocurre."""
    t = pd.crosstab(df["generation"], df["dropout.semester"])
    out = pd.DataFrame({"n": t.sum(axis=1)})
    for code, label in MOMENTOS.items():
        out[label] = t.get(code, 0) / out["n"]
    out["tasa_total"] = out[list(MOMENTOS.values())].sum(axis=1)
    return out


# ---------------------------------------------------------------- faltantes
def missing_by_cohort(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """% de registros ausentes por variable y cohorte: distingue faltantes estructurales."""
    return (df[cols].apply(is_missing).groupby(df["generation"]).mean() * 100).T.round(1)


def missing_vs_target(df: pd.DataFrame, cols: list[str], min_pct: float = 0.5) -> pd.DataFrame:
    """¿El dato faltante se asocia con desertar? Tasa con y sin dato, y prueba chi-cuadrado.

    Se omiten las variables con menos de ``min_pct`` % de ausentes (tasas sin soporte).
    """
    rows = []
    for c in cols:
        miss = is_missing(df[c])
        if miss.all() or 100 * miss.mean() < min_pct:
            continue
        tab = pd.crosstab(miss, df[TARGET])
        _, p, _, _ = stats.chi2_contingency(tab)
        rows.append({"variable": c, "pct_ausente": 100 * miss.mean(),
                     "tasa_con_dato": df.loc[~miss, TARGET].mean(),
                     "tasa_sin_dato": df.loc[miss, TARGET].mean(), "p_chi2": p})
    out = pd.DataFrame(rows)
    out["diferencia_pp"] = 100 * (out["tasa_sin_dato"] - out["tasa_con_dato"])
    return out.sort_values("pct_ausente", ascending=False)


# --------------------------------------------------------- relación con y
def rate_by_category(df: pd.DataFrame, col: str, min_n: int = 100) -> pd.DataFrame:
    """Tasa de deserción por categoría, con intervalo de Wilson al 95 %."""
    g = df.groupby(df[col].fillna(NO_INFO))[TARGET].agg(n="size", desertores="sum")
    g = g[g["n"] >= min_n].copy()
    g["tasa"] = g["desertores"] / g["n"]
    g["ic_inf"], g["ic_sup"] = wilson(g["desertores"], g["n"])
    return g.sort_values("tasa")


def rate_by_quantile(df: pd.DataFrame, col: str, q: int = 10) -> pd.DataFrame:
    """Tasa de deserción por cuantiles de una variable numérica (los NaN se reportan aparte)."""
    s = pd.to_numeric(df[col], errors="coerce")
    bins = pd.qcut(s, q=q, duplicates="drop")
    g = df.groupby(bins, observed=True)[TARGET].agg(n="size", desertores="sum")
    g["centro"] = s.groupby(bins, observed=True).median()
    if s.isna().any():
        na = df.loc[s.isna(), TARGET]
        g.loc["Sin dato"] = [len(na), na.sum(), np.nan]
    g["tasa"] = g["desertores"] / g["n"]
    g["ic_inf"], g["ic_sup"] = wilson(g["desertores"], g["n"])
    return g


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér corregida por sesgo (Bergsma, 2013)."""
    tab = pd.crosstab(x, y)
    if min(tab.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k_c - 1, r_c - 1))))


def univariate_association(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    """Fuerza de la asociación de cada variable, por separado, con la deserción.

    - Numéricas: AUC univariada (0,5 = sin señal; < 0,5 = a mayor valor, menor riesgo) y
      diferencia de medias estandarizada (d de Cohen) entre desertores y retenidos.
    - Categóricas: V de Cramér. Para tener una escala común se reporta además la AUC que
      se obtendría ordenando por la tasa de deserción de cada categoría (descriptiva, en muestra).
    """
    y = df[TARGET].to_numpy()
    rows = []
    for c in num:
        s = pd.to_numeric(df[c], errors="coerce")
        ok = s.notna().to_numpy()
        auc = roc_auc_score(y[ok], s[ok])
        a, b = s[y == 1].dropna(), s[y == 0].dropna()
        sd = np.sqrt((a.var() + b.var()) / 2)
        d = (a.mean() - b.mean()) / sd if sd > 0 else np.nan
        rows.append({"variable": c, "tipo": "numérica", "pct_ausente": 100 * (~ok).mean(),
                     "auc": auc, "fuerza_auc": abs(auc - 0.5) * 2, "d_cohen": d,
                     "v_cramer": np.nan})
    for c in cat:
        s = df[c].fillna(NO_INFO)
        score = s.map(df.groupby(s)[TARGET].mean())
        auc = roc_auc_score(y, score)
        rows.append({"variable": c, "tipo": "categórica", "pct_ausente": 100 * is_missing(s).mean(),
                     "auc": auc, "fuerza_auc": abs(auc - 0.5) * 2, "d_cohen": np.nan,
                     "v_cramer": cramers_v(s, df[TARGET])})
    return pd.DataFrame(rows).sort_values("fuerza_auc", ascending=False).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Resumen de las numéricas: rango, cuantiles y mediana según el desenlace."""
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({"variable": c, "n_validos": int(s.notna().sum()), "min": s.min(),
                     "p01": s.quantile(0.01), "mediana": s.median(), "p99": s.quantile(0.99),
                     "max": s.max(), "mediana_retenidos": s[df[TARGET] == 0].median(),
                     "mediana_desertores": s[df[TARGET] == 1].median()})
    return pd.DataFrame(rows)


def numeric_correlation(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Correlación de Spearman entre numéricas (robusta a asimetrías y valores extremos)."""
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
