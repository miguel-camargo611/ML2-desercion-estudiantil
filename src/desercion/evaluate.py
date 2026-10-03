"""Métricas, validación cruzada anidada y pruebas estadísticas de comparación.

Esquema por modelo (sin fuga de información):

    pliegue externo k (StratifiedGroupKFold, 5)
      ├─ entrenamiento_k ──► GridSearchCV (StratifiedGroupKFold interno, 3; métrica PR-AUC)
      │                       └─ Pipeline: imputar/escalar/one-hot ► (SMOTE) ► clasificador
      │                  ──► umbral: predicciones fuera de pliegue (3) dentro de entrenamiento_k,
      │                       se elige el que maximiza F2
      └─ validación_k   ──► solo predict: métricas con el umbral ya fijado
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict

from desercion import config
from desercion.models import ModelSpec, build_pipeline, params_to_text

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------ utilidades
def scores_of(estimator, X) -> np.ndarray:
    """Puntaje continuo de la clase positiva (probabilidad o función de decisión)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def best_threshold(y_true: np.ndarray, scores: np.ndarray, beta: float) -> float:
    """Umbral que maximiza F-beta sobre predicciones fuera de pliegue."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    denom = beta**2 * precision + recall
    fbeta = np.divide((1 + beta**2) * precision * recall, denom, out=np.zeros_like(denom),
                      where=denom > 0)
    return float(thresholds[int(np.argmax(fbeta))])


def binary_metrics(y: np.ndarray, scores: np.ndarray, pred: np.ndarray, beta: float) -> dict:
    return {
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores) if np.unique(scores).size > 1 else 0.5,
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        f"f{beta:g}": fbeta_score(y, pred, beta=beta, zero_division=0),
        "f1_desercion": f1_score(y, pred, zero_division=0),
        "f1_retenido": f1_score(1 - y, 1 - pred, zero_division=0),
        "f1_ponderado": f1_score(y, pred, average="weighted", zero_division=0),
        "bal_accuracy": balanced_accuracy_score(y, pred),
        "tasa_alerta": float(np.mean(pred)),
    }


def _cv(n_splits: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ------------------------------------------------------------------ CV anidada
def nested_cv(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
              num: list[str], cat: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config.load_config()
    seed = config.seed()
    beta = cfg["metrics"]["threshold_beta"]
    outer = _cv(cfg["cv"]["outer_folds"], seed)
    fold_rows, oof_rows = [], []

    for k, (tr, va) in enumerate(outer.split(X, y, groups)):
        t0 = time.perf_counter()
        X_tr, y_tr, g_tr = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va].to_numpy()
        pipe = build_pipeline(spec, num, cat, seed=seed)

        if spec.grid:
            search = GridSearchCV(pipe, spec.grid, scoring=cfg["metrics"]["tuning"],
                                  cv=_cv(cfg["cv"]["inner_folds"], seed + 1),
                                  n_jobs=spec.search_n_jobs, refit=True, error_score="raise")
            search.fit(X_tr, y_tr, groups=g_tr)
            model, params = search.best_estimator_, search.best_params_
        else:
            model, params = pipe.fit(X_tr, y_tr), {}

        # Umbral: predicciones fuera de pliegue dentro del entrenamiento externo (sin tocar validación).
        method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
        oof_tr = cross_val_predict(clone(model), X_tr, y_tr, groups=g_tr,
                                   cv=_cv(cfg["cv"]["threshold_folds"], seed + 2), method=method)
        oof_tr = oof_tr[:, 1] if oof_tr.ndim == 2 else oof_tr
        thr = best_threshold(y_tr.to_numpy(), oof_tr, beta)

        s_va = scores_of(model, X_va)
        pred = (s_va >= thr).astype(int)
        m = binary_metrics(y_va, s_va, pred, beta)
        m_default = binary_metrics(y_va, s_va, model.predict(X_va).astype(int), beta)
        fold_rows.append({
            "modelo": spec.name, "familia": spec.family, "pliegue": k, **m,
            "recall_umbral_def": m_default["recall"], "f1_ponderado_umbral_def": m_default["f1_ponderado"],
            "umbral": thr, "params": params_to_text(params), "segundos": time.perf_counter() - t0,
        })
        oof_rows.append(pd.DataFrame({"row_id": X_va.index, "pliegue": k, "y": y_va,
                                      "score": s_va, "pred": pred, "modelo": spec.name}))
        log.info("%s | pliegue %d | PR-AUC %.3f | F2 %.3f | %.0fs", spec.name, k, m["pr_auc"],
                 m[f"f{beta:g}"], fold_rows[-1]["segundos"])

    return pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True)


def summarize(folds: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Media ± desviación estándar de cada métrica a través de los pliegues externos."""
    metrics = metrics or ["pr_auc", "roc_auc", "recall", "precision", "f2", "f1_desercion",
                          "f1_ponderado", "bal_accuracy", "tasa_alerta", "recall_umbral_def",
                          "f1_ponderado_umbral_def"]
    agg = folds.groupby("modelo", sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.sort_values("pr_auc_mean", ascending=False)


def capture_at_top(oof: pd.DataFrame, fractions=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Recall y precisión si se alerta solo al k % de estudiantes con mayor riesgo (por pliegue)."""
    rows = []
    for (model, fold), d in oof.groupby(["modelo", "pliegue"]):
        d = d.sort_values("score", ascending=False)
        for frac in fractions:
            top = d.head(int(round(frac * len(d))))
            rows.append({"modelo": model, "pliegue": fold, "fraccion_alertada": frac,
                         "recall": top["y"].sum() / d["y"].sum(), "precision": top["y"].mean(),
                         "lift": top["y"].mean() / d["y"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- pruebas estadísticas
def compare_models(folds: pd.DataFrame, a: str, b: str, metric: str, n_train: int,
                   n_test: int) -> dict[str, Any]:
    """Compara dos modelos sobre los mismos pliegues externos.

    - t pareada clásica (la que usó el artículo vía STAC),
    - Wilcoxon de rangos con signo (no paramétrica),
    - t corregida de Nadeau & Bengio (2003), que ajusta la varianza por el solapamiento
      de los conjuntos de entrenamiento entre pliegues.
    """
    da = folds[folds.modelo == a].sort_values("pliegue")[metric].to_numpy()
    db = folds[folds.modelo == b].sort_values("pliegue")[metric].to_numpy()
    d = da - db
    k = d.size
    t_p, p_t = stats.ttest_rel(da, db)
    try:
        _, p_w = stats.wilcoxon(da, db)
    except ValueError:
        p_w = float("nan")
    var = np.var(d, ddof=1)
    t_nb = d.mean() / np.sqrt((1 / k + n_test / n_train) * var) if var > 0 else np.inf
    p_nb = 2 * stats.t.sf(abs(t_nb), df=k - 1)
    return {"modelo_a": a, "modelo_b": b, "metrica": metric, "media_a": da.mean(),
            "media_b": db.mean(), "diferencia": d.mean(), "t_pareada": t_p, "p_t_pareada": p_t,
            "p_wilcoxon": p_w, "t_nadeau_bengio": t_nb, "p_nadeau_bengio": p_nb}
