"""Interpretación preliminar: SHAP por pliegue (importancia + estabilidad) y análisis de errores."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedGroupKFold

from desercion import config
from desercion.data import GROUP, TARGET
from desercion.features import feature_lists
from desercion.models import ModelSpec, tuned_pipeline

log = logging.getLogger(__name__)


def original_variable(feature: str, originals: list[str]) -> str:
    """Mapea una columna transformada (one-hot o indicador de faltante) a su variable original."""
    if feature.startswith("missingindicator_"):
        feature = feature.removeprefix("missingindicator_")
    matches = [o for o in originals if feature == o or feature.startswith(o + "_")]
    return max(matches, key=len) if matches else feature


def _shap_matrix(explainer, X) -> np.ndarray:
    values = explainer.shap_values(X)
    if isinstance(values, list):  # API antigua: una matriz por clase
        values = values[1]
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


def shap_by_fold(dev: pd.DataFrame, spec: ModelSpec, params: dict,
                 n_sample: int = 2000) -> tuple[pd.DataFrame, dict]:
    """Ajusta el modelo en cada pliegue externo y calcula |SHAP| medio en su validación.

    Devuelve la importancia agregada por variable original en cada pliegue (para medir
    estabilidad) y los insumos del último pliegue para el gráfico *beeswarm*.
    """
    seed = config.seed()
    num, cat = feature_lists()
    originals = num + cat
    X, y, g = dev[originals], dev[TARGET], dev[GROUP]
    outer = StratifiedGroupKFold(n_splits=config.load_config()["cv"]["outer_folds"], shuffle=True,
                                 random_state=seed)
    rng = np.random.default_rng(seed)
    rows, last = [], {}
    for k, (tr, va) in enumerate(outer.split(X, y, g)):
        pipe = tuned_pipeline(spec, params, num, cat).fit(X.iloc[tr], y.iloc[tr])
        va_s = rng.choice(va, size=min(n_sample, len(va)), replace=False)
        Xv = pipe.named_steps["prep"].transform(X.iloc[va_s])
        names = list(pipe.named_steps["prep"].get_feature_names_out())
        clf = pipe.named_steps["clf"]
        sv = _shap_matrix(shap.TreeExplainer(getattr(clf, "model_", clf)), Xv)
        # Por aditividad de SHAP, la contribución de una variable original es la suma (con signo)
        # de las de sus columnas transformadas; luego se promedia el valor absoluto.
        per_var = pd.DataFrame(sv, columns=names).T.groupby(
            [original_variable(f, originals) for f in names]).sum().T
        rows.append(per_var.abs().mean(axis=0).rename(k))
        last = {"shap": sv, "X": pd.DataFrame(np.asarray(Xv), columns=names)}
        log.info("SHAP pliegue %d listo", k)
    table = pd.concat(rows, axis=1)
    table.columns = [f"pliegue_{c}" for c in table.columns]
    table["media"] = table.mean(axis=1)
    table["sd"] = table.filter(like="pliegue_").std(axis=1)
    ranks = table.filter(like="pliegue_").rank(ascending=False)
    table["rango_medio"] = ranks.mean(axis=1)
    table["rango_min"] = ranks.min(axis=1)
    table["rango_max"] = ranks.max(axis=1)
    return table.sort_values("media", ascending=False), last


def error_analysis(oof: pd.DataFrame, dev: pd.DataFrame, segments: list[str]) -> pd.DataFrame:
    """Recall y tasa de alerta del mejor modelo por segmento (predicciones fuera de pliegue)."""
    df = oof.merge(dev[segments], left_on="row_id", right_index=True)
    out = []
    for seg in segments:
        grp = df.groupby(seg)
        t = pd.DataFrame({
            "n": grp.size(),
            "desertores": grp["y"].sum(),
            "tasa_desercion": grp["y"].mean(),
            "recall": grp.apply(lambda d: d.loc[d.y == 1, "pred"].mean(), include_groups=False),
            "falsos_positivos_pct": grp.apply(lambda d: d.loc[d.y == 0, "pred"].mean(),
                                              include_groups=False),
        })
        t = t[t["desertores"] >= 30].reset_index().rename(columns={seg: "categoria"})
        t.insert(0, "segmento", seg)
        out.append(t)
    return pd.concat(out, ignore_index=True)
