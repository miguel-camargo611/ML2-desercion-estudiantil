"""Modelo final: ajuste con todo el conjunto de desarrollo, evaluación única en prueba y artefactos.

Protocolo (el mismo de la CV anidada, ahora sobre todo el desarrollo):

    desarrollo ──► GridSearchCV (StratifiedGroupKFold, 3; PR-AUC) ──► hiperparámetros
               ──► umbral F2 con predicciones fuera de pliegue (3)
               ──► ajuste final con todo el desarrollo
    prueba     ──► solo predict, una vez: métricas, captura, calibración e intervalos bootstrap

El conjunto de prueba no interviene en ninguna decisión. Los artefactos siguen el contrato
visto en clase: modelo serializado, metadatos (orden de variables, contrato de entradas,
umbral, versión, métricas, limitaciones) y una predicción de referencia para verificar la
paridad entre entornos.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict

from desercion import config
from desercion.data import GROUP, TARGET
from desercion.evaluate import best_threshold, binary_metrics
from desercion.features import KEPT_ENGINEERED, VAR_LABELS, add_engineered, feature_lists
from desercion.models import ModelSpec, build_pipeline, get_specs, params_to_text

log = logging.getLogger(__name__)


def final_spec() -> ModelSpec:
    """Modelo final definido en ``configs/config.yaml`` (nombre y numéricas sin marca de faltante)."""
    cfg = config.load_config()["final_model"]
    return replace(get_specs()[cfg["name"]], no_indicator=tuple(cfg["no_indicator"]))


def _cv(n: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)


def fit_final(dev: pd.DataFrame) -> dict:
    """Elige hiperparámetros y umbral con todo el desarrollo y ajusta el modelo final."""
    cfg, seed = config.load_config(), config.seed()
    spec = final_spec()
    num, cat = feature_lists()
    X, y, g = dev[num + cat], dev[TARGET], dev[GROUP]
    search = GridSearchCV(build_pipeline(spec, num, cat), spec.grid,
                          scoring=cfg["metrics"]["tuning"], cv=_cv(cfg["cv"]["inner_folds"], seed + 1),
                          n_jobs=spec.search_n_jobs, refit=True, error_score="raise")
    search.fit(X, y, groups=g)
    model = search.best_estimator_
    oof = cross_val_predict(clone(model), X, y, groups=g, method="predict_proba",
                            cv=_cv(cfg["cv"]["threshold_folds"], seed + 2))[:, 1]
    thr = best_threshold(y.to_numpy(), oof, cfg["metrics"]["threshold_beta"])
    log.info("Modelo final: %s | umbral %.4f | PR-AUC CV interna %.3f", search.best_params_, thr,
             search.best_score_)
    return {"model": model, "params": search.best_params_, "threshold": thr,
            "cv_pr_auc": search.best_score_, "oof_dev": oof, "num": num, "cat": cat}


def _capture(y: np.ndarray, s: np.ndarray, frac: float) -> tuple[float, float]:
    top = np.argsort(-s, kind="stable")[: int(round(frac * len(s)))]
    return float(y[top].sum() / y.sum()), float(y[top].mean())


def evaluate_test(fit: dict, test: pd.DataFrame, n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluación única en prueba: métricas con el umbral fijado, captura e intervalos al 95 %."""
    cfg, seed = config.load_config(), config.seed()
    beta = cfg["metrics"]["threshold_beta"]
    X = test[fit["num"] + fit["cat"]]
    y, s = test[TARGET].to_numpy(), fit["model"].predict_proba(X)[:, 1]
    pred = (s >= fit["threshold"]).astype(int)
    m = binary_metrics(y, s, pred, beta)
    m.update({"n": len(y), "desertores": int(y.sum()), "prevalencia": y.mean(),
              "brier": brier_score_loss(y, s), "brier_tasa_base": y.mean() * (1 - y.mean()),
              "prob_media": s.mean(), "umbral": fit["threshold"]})
    for frac in (0.10, 0.20):
        m[f"captura_top{int(frac * 100)}"], m[f"precision_top{int(frac * 100)}"] = _capture(y, s, frac)

    # Intervalos bootstrap (percentil) de las métricas principales.
    rng = np.random.default_rng(seed)
    boot = {"pr_auc": [], "roc_auc": [], "recall": [], "precision": [], "captura_top20": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        bm = binary_metrics(y[i], s[i], pred[i], beta)
        for k in ("pr_auc", "roc_auc", "recall", "precision"):
            boot[k].append(bm[k])
        boot["captura_top20"].append(_capture(y[i], s[i], 0.20)[0])
    rows = [{"metrica": k, "valor": m[k], "ic95_bajo": np.percentile(v, 2.5),
             "ic95_alto": np.percentile(v, 97.5)} for k, v in boot.items()]
    rows += [{"metrica": k, "valor": v} for k, v in m.items() if k not in boot]

    by_cohort = []
    for gen, d in test.assign(score=s, pred=pred).groupby("generation"):
        yy, ss = d[TARGET].to_numpy(), d["score"].to_numpy()
        by_cohort.append({"cohorte": gen, "n": len(d), "prevalencia": yy.mean(),
                          "pr_auc": average_precision_score(yy, ss), "roc_auc": roc_auc_score(yy, ss),
                          "recall": float(d.loc[d[TARGET] == 1, "pred"].mean()),
                          "prob_media": ss.mean()})
    return pd.DataFrame(rows), pd.DataFrame(by_cohort)


# ------------------------------------------------------------------ SHAP por variable
def shap_groups(model, num: list[str], cat: list[str]) -> dict[str, str]:
    """Columna transformada -> variable original (une one-hot y marcas de faltante)."""
    names = model.named_steps["prep"].get_feature_names_out()
    out = {}
    for n in names:
        if n in num:
            out[n] = n
        elif n.startswith("missingindicator_"):
            out[n] = n.removeprefix("missingindicator_")
        else:
            out[n] = max((c for c in cat if n.startswith(c + "_")), key=len)
    return out


def shap_by_variable(model, X: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Valores SHAP (log-odds) sumados por variable original, una fila por estudiante."""
    Xt = model.named_steps["prep"].transform(X)
    values = np.asarray(shap.TreeExplainer(model.named_steps["clf"]).shap_values(Xt))
    if values.ndim == 3:
        values = values[..., 1] if values.shape[-1] == 2 else values[1]
    names = model.named_steps["prep"].get_feature_names_out()
    df = pd.DataFrame(values, columns=names, index=X.index)
    return df.T.groupby(pd.Series(groups)).sum().T


# ------------------------------------------------------------------ artefactos
def _input_contract(dev: pd.DataFrame, num: list[str], cat: list[str]) -> dict:
    contract = {}
    for c in num:
        if c in KEPT_ENGINEERED:
            continue
        s = dev[c]
        contract[c] = {"type": "numeric", "label": VAR_LABELS.get(c, c),
                       "minimum": float(s.min()), "maximum": float(s.max()),
                       "default": float(s.median()),
                       "integer": bool(np.allclose(s.dropna() % 1, 0)),
                       # Solo se admite "sin dato" donde faltar es habitual (> 1 %), p. ej. el
                       # examen presencial de quien lo presentó en línea; no por casos aislados.
                       "allow_missing": bool(s.isna().mean() > 0.01)}
    for c in cat:
        counts = dev[c].value_counts()
        contract[c] = {"type": "categorical", "label": VAR_LABELS.get(c, c),
                       "categories": counts.index.tolist(), "default": counts.index[0]}
    return contract


def export_artifacts(fit: dict, dev: pd.DataFrame, test_metrics: pd.DataFrame, out: Path) -> None:
    """Escribe model.joblib, model_metadata.json y reference_prediction.json en ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    model, num, cat = fit["model"], fit["num"], fit["cat"]
    groups = shap_groups(model, num, cat)
    sample = dev.sample(n=min(3000, len(dev)), random_state=config.seed())
    imp = shap_by_variable(model, sample[num + cat], groups).abs().mean().sort_values(ascending=False)
    contract = _input_contract(dev, num, cat)
    tm = test_metrics.set_index("metrica")["valor"]
    import imblearn
    import lightgbm

    metadata = {
        "model_name": "LightGBM — riesgo de deserción al terminar el 1.er semestre",
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": "Student Dropout Dataset, Tec de Monterrey (cohortes AD19–AD20, pregrado)",
        "target": "desercion (1 = ya no está matriculado al cerrar el primer año)",
        "positive_class": 1,
        "feature_order": num + cat,
        "input_features": list(contract),
        "engineered": {"brecha.promedio": "average.first.period - PNA"},
        "input_contract": contract,
        "decision_threshold": fit["threshold"],
        "hyperparameters": params_to_text(fit["params"]),
        "metrics": {k: round(float(tm[k]), 4) for k in
                    ("pr_auc", "roc_auc", "recall", "precision", "captura_top20", "prevalencia")},
        "risk_percentiles": np.percentile(fit["oof_dev"], np.arange(1, 100)).round(6).tolist(),
        "global_importance": [{"feature": k, "label": VAR_LABELS.get(k, k), "importance": float(v)}
                              for k, v in imp.items()],
        "shap_groups": groups,
        "limitations": [
            "Demostración académica: no usar para decisiones sobre estudiantes reales sin "
            "validación institucional.",
            "La señal es débil (ROC-AUC ≈ 0,7): la alerta prioriza a quién acompañar, no "
            "diagnostica a un estudiante.",
            "Entrenado con AD19–AD20; en una cohorte nueva el desempeño baja algo y el nivel "
            "de las probabilidades puede correrse (validación temporal AD19 → AD20).",
            "La explicación SHAP describe cómo decide el modelo; no implica causalidad.",
            "Los rangos de las entradas son los del entrenamiento: valores fuera de ellos no "
            "están respaldados por los datos.",
        ],
        "runtime": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                    "imbalanced_learn": imblearn.__version__, "lightgbm": lightgbm.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__, "shap": shap.__version__},
    }
    joblib.dump(model, out / "model.joblib")
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    # Perfil ficticio (medianas y modas del desarrollo): nunca un estudiante real.
    profile = {c: spec["default"] for c, spec in contract.items()}
    frame = add_engineered(pd.DataFrame([profile]))[num + cat]
    prob = float(model.predict_proba(frame)[0, 1])
    reference = {"descripcion": "Perfil ficticio: mediana o moda de cada variable en desarrollo.",
                 "input": profile, "probability": prob,
                 "prediction": int(prob >= fit["threshold"]), "tolerance": 1e-6}
    (out / "reference_prediction.json").write_text(json.dumps(reference, indent=2,
                                                              ensure_ascii=False), encoding="utf-8")
    log.info("Artefactos en %s (referencia: p = %.4f)", out, prob)


# ---------------------------------------------- lectura de la evaluación en prueba
# Estas funciones solo DESCRIBEN el resultado ya obtenido en prueba (no se toma ninguna
# decisión de modelado con ellas): niveles de riesgo, errores y dirección de cada variable.
def risk_tiers(score: np.ndarray, metadata: dict) -> np.ndarray:
    """Mismos niveles que la app: Alto = 20 % de mayor riesgo; Medio = supera el umbral F2."""
    p80, thr = metadata["risk_percentiles"][79], metadata["decision_threshold"]
    return np.where(score >= p80, "Alto", np.where(score >= thr, "Medio", "Bajo"))


def tier_summary(y: np.ndarray, score: np.ndarray, metadata: dict) -> pd.DataFrame:
    tiers = risk_tiers(score, metadata)
    rows = []
    for t in ("Alto", "Medio", "Bajo"):
        m = tiers == t
        rows.append({"nivel": t, "estudiantes": int(m.sum()), "pct_estudiantes": m.mean(),
                     "tasa_desercion": y[m].mean() if m.any() else np.nan,
                     "desertores": int(y[m].sum()), "pct_desertores": y[m].sum() / y.sum()})
    return pd.DataFrame(rows)


def confusion_summary(y: np.ndarray, score: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = score >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return pd.DataFrame([{"vp": tp, "fp": fp, "fn": fn, "vn": tn, "alertas": tp + fp,
                          "recall": tp / (tp + fn), "precision": tp / (tp + fp),
                          "alertas_por_desertor_detectado": (tp + fp) / tp}])


def errors_by_group(test: pd.DataFrame, score: np.ndarray, threshold: float,
                    cols: list[str], min_dropouts: int = 15) -> pd.DataFrame:
    """Recall y tasa de alerta por grupo (solo grupos con suficientes desertores)."""
    d = test.assign(score=score, pred=(score >= threshold).astype(int))
    rows = []
    for col in cols:
        for value, g in d.groupby(col):
            if g[TARGET].sum() < min_dropouts:
                continue
            rows.append({"variable": col, "grupo": value, "n": len(g),
                         "desertores": int(g[TARGET].sum()), "tasa_desercion": g[TARGET].mean(),
                         "recall": g.loc[g[TARGET] == 1, "pred"].mean(),
                         "tasa_alerta": g["pred"].mean()})
    return pd.DataFrame(rows)


def shap_direction(model, X: pd.DataFrame, groups: dict[str, str], num: list[str]) -> pd.DataFrame:
    """Importancia (media |SHAP|) y dirección: correlación de Spearman entre valor y SHAP."""
    sv = shap_by_variable(model, X, groups)
    rows = []
    for var in sv.abs().mean().sort_values(ascending=False).index:
        rho = (pd.Series(sv[var]).corr(X[var].astype(float), method="spearman")
               if var in num else np.nan)
        rows.append({"variable": var, "etiqueta": VAR_LABELS.get(var, var),
                     "shap_abs_medio": float(sv[var].abs().mean()), "spearman_valor_shap": rho})
    return pd.DataFrame(rows)


def capture_curve(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """% de desertores detectados al alertar al k % de estudiantes con mayor riesgo."""
    order = np.argsort(-score, kind="stable")
    captured = np.cumsum(y[order]) / y.sum()
    frac = np.arange(1, len(y) + 1) / len(y)
    return pd.DataFrame({"fraccion_alertada": frac, "fraccion_desertores": captured})


def calibration_deciles(y: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    from desercion.eda import wilson

    d = pd.DataFrame({"y": y, "p": score})
    d["bin"] = pd.qcut(d["p"], bins, labels=False, duplicates="drop")
    c = d.groupby("bin").agg(prob_media=("p", "mean"), tasa_observada=("y", "mean"),
                             desertores=("y", "sum"), n=("y", "size")).reset_index()
    c["ic_bajo"], c["ic_alto"] = wilson(c["desertores"], c["n"])
    return c
