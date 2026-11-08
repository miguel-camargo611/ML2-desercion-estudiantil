"""Selección de variables y experimentos que reproducen (y ponen a prueba) decisiones del artículo.

1. ``feature_engineering_check`` – ¿aportan las variables construidas?
2. ``feature_selection``         – importancia por permutación dentro de cada partición.
3. ``resampling_comparison``     – SMOTE vs ADASYN con un árbol de decisión de control (artículo).
4. ``leakage_demo``              – protocolo del artículo (SMOTE antes de partir) vs Pipeline.
5. ``isolation_forest``          – ¿son los desertores "anomalías"? (etapas 1 y 2 del artículo).
6. ``vif``                       – multicolinealidad que llevó al artículo a descartar la RL.
7. ``validation_curve``          – desempeño en entrenamiento y validación según un hiperparámetro.
8. ``computational_cost``        – tiempos de búsqueda, ajuste y predicción, y tamaño del modelo.
9. Sensibilidad del modelo final – marcas de faltante, validación temporal y calibración.
"""

from __future__ import annotations

import logging
import os
import pickle
import platform
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

from desercion import config
from desercion.data import GROUP, TARGET
from desercion.evaluate import best_threshold, binary_metrics, nested_cv
from desercion.features import ENGINEERED, feature_lists, make_preprocessor
from desercion.models import (
    ModelSpec,
    build_pipeline,
    params_from_text,
    params_to_text,
    tuned_pipeline,
)

log = logging.getLogger(__name__)

# Configuración fija de LightGBM para los experimentos (cercana a la elegida por la CV anidada).
LGBM_FIXED = dict(n_estimators=400, num_leaves=8, learning_rate=0.05, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def _outer(seed: int, n: int = 5) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)


def _lgbm_spec(seed: int, **extra) -> ModelSpec:
    return ModelSpec("LightGBM", "Boosting",
                     LGBMClassifier(random_state=seed, n_jobs=-1, **{**LGBM_FIXED, **extra}))


def _logit_spec(seed: int) -> ModelSpec:
    return ModelSpec("Regresión logística", "", LogisticRegression(C=0.01, max_iter=5000,
                                                                     random_state=seed))


def _cv_pr_auc(dev: pd.DataFrame, num: list[str], cat: list[str], spec: ModelSpec,
               seed: int) -> list[float]:
    X, y, g = dev[num + cat], dev[TARGET], dev[GROUP]
    out = []
    for tr, va in _outer(seed).split(X, y, g):
        pipe = build_pipeline(spec, num, cat).fit(X.iloc[tr], y.iloc[tr])
        out.append(average_precision_score(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1]))
    return out


# --------------------------------------------------- 1. Variables construidas
def feature_engineering_check(dev: pd.DataFrame) -> pd.DataFrame:
    """PR-AUC por pliegue sin y con cada variable construida (LightGBM y regresión logística)."""
    seed = config.seed()
    variants = {"Sin variables construidas": list(ENGINEERED)}
    variants.update({f"+ {v}": [o for o in ENGINEERED if o != v] for v in ENGINEERED})
    variants["+ ambas"] = []
    rows = []
    for name, drop in variants.items():
        num, cat = feature_lists(engineered=list(ENGINEERED), selected=False, drop=drop)
        for spec in (_lgbm_spec(seed), _logit_spec(seed)):
            for k, v in enumerate(_cv_pr_auc(dev, num, cat, spec, seed)):
                rows.append({"variante": name, "modelo": spec.name, "pliegue": k, "pr_auc": v})
        log.info("Variables construidas | %s listo", name)
    return pd.DataFrame(rows)


# ----------------------------------------------------- 2. Selección de variables
def feature_selection(dev: pd.DataFrame, n_repeats: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selección por importancia de permutación, ajustada dentro de cada partición.

    En cada pliegue externo, la importancia (caída del PR-AUC al permutar la variable) se mide
    con una validación cruzada interna sobre el entrenamiento; se conservan las variables con
    importancia positiva. Luego se compara, en la validación externa, el modelo con todas las
    variables y con las seleccionadas. Devuelve (importancias por pliegue, PR-AUC por pliegue).
    """
    seed = config.seed()
    num, cat = feature_lists(selected=False)
    cols = num + cat
    X, y, g = dev[cols], dev[TARGET], dev[GROUP]
    imp_rows, score_rows = [], []
    for k, (tr, va) in enumerate(_outer(seed).split(X, y, g)):
        X_tr, y_tr, g_tr = X.iloc[tr], y.iloc[tr], g.iloc[tr]
        inner = []
        for itr, iva in _outer(seed + 3, 3).split(X_tr, y_tr, g_tr):
            pipe = build_pipeline(_lgbm_spec(seed), num, cat).fit(X_tr.iloc[itr], y_tr.iloc[itr])
            r = permutation_importance(pipe, X_tr.iloc[iva], y_tr.iloc[iva],
                                       scoring="average_precision", n_repeats=n_repeats,
                                       random_state=seed, n_jobs=1)
            inner.append(r.importances_mean)
        imp = pd.Series(np.mean(inner, axis=0), index=cols)
        keep = imp[imp > 0].index.tolist()
        for c in cols:
            imp_rows.append({"variable": c, "pliegue": k, "importancia": imp[c],
                             "seleccionada": c in keep})
        k_num, k_cat = [c for c in num if c in keep], [c for c in cat if c in keep]
        for label, (n_, c_) in {"Todas las candidatas": (num, cat),
                                "Seleccionadas": (k_num, k_cat)}.items():
            for spec in (_lgbm_spec(seed), _logit_spec(seed)):
                pipe = build_pipeline(spec, n_, c_).fit(X_tr[n_ + c_], y_tr)
                s = pipe.predict_proba(X.iloc[va][n_ + c_])[:, 1]
                score_rows.append({"conjunto": label, "modelo": spec.name, "pliegue": k,
                                   "n_variables": len(n_) + len(c_),
                                   "pr_auc": average_precision_score(y.iloc[va], s)})
        log.info("Selección | pliegue %d: %d de %d variables", k, len(keep), len(cols))
    return pd.DataFrame(imp_rows), pd.DataFrame(score_rows)


def selection_check(dev: pd.DataFrame, drop: list[str]) -> pd.DataFrame:
    """Verifica en particiones nuevas (otra semilla) que quitar ``drop`` no empeora el PR-AUC."""
    seed = config.seed() + 10
    rows = []
    for label, d in {"Todas las candidatas": [], "Seleccionadas": drop}.items():
        num, cat = feature_lists(selected=False, drop=d)
        for spec in (_lgbm_spec(seed), _logit_spec(seed)):
            for k, v in enumerate(_cv_pr_auc(dev, num, cat, spec, seed)):
                rows.append({"conjunto": label, "modelo": spec.name, "pliegue": k,
                             "n_variables": len(num) + len(cat), "pr_auc": v})
    return pd.DataFrame(rows)


# -------------------------------------------------------------- 3. SMOTE vs ADASYN
def resampling_comparison(dev: pd.DataFrame) -> pd.DataFrame:
    """Réplica de la comparación del artículo: árbol de control sin balanceo, SMOTE y ADASYN."""
    seed = config.seed()
    num, cat = feature_lists()
    X, y, g = dev[num + cat], dev[TARGET], dev[GROUP]
    control = ModelSpec("Árbol de control", "", DecisionTreeClassifier(random_state=seed))
    variants = {"Sin balanceo": "passthrough", "SMOTE": SMOTE(random_state=seed),
                "ADASYN": ADASYN(random_state=seed)}
    rows = []
    for name, sampler in variants.items():
        for k, (tr, va) in enumerate(_outer(seed).split(X, y, g)):
            pipe = build_pipeline(control, num, cat, sampler=sampler).fit(X.iloc[tr], y.iloc[tr])
            yv, pred = y.iloc[va].to_numpy(), pipe.predict(X.iloc[va])
            rows.append({"tecnica": name, "pliegue": k,
                         "f1_retenido": f1_score(1 - yv, 1 - pred),
                         "f1_desercion": f1_score(yv, pred),
                         "f1_ponderado": f1_score(yv, pred, average="weighted"),
                         "recall_desercion": recall_score(yv, pred)})
        log.info("Balanceo %s listo", name)
    return pd.DataFrame(rows)


# -------------------------------------------------- 4. Demostración de fuga (SMOTE)
def leakage_demo(dev: pd.DataFrame) -> pd.DataFrame:
    """Compara el protocolo del artículo con el Pipeline sin fuga, mismo modelo e hiperparámetros.

    - "Artículo (SMOTE antes de partir)": se preprocesa y sobremuestrea TODO el conjunto y luego
      se hace la CV: los pliegues de validación contienen puntos sintéticos interpolados a partir
      de vecinos que están en entrenamiento.
    - "Pipeline (SMOTE dentro del pliegue)": el remuestreo solo ve el entrenamiento de cada pliegue.
    """
    seed = config.seed()
    num, cat = feature_lists()
    X, y = dev[num + cat], dev[TARGET].to_numpy()
    clf = LGBMClassifier(random_state=seed, n_jobs=-1, **LGBM_FIXED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []

    def _record(protocol, k, yv, pred, proba):
        rows.append({"protocolo": protocol, "pliegue": k,
                     "f1_ponderado": f1_score(yv, pred, average="weighted"),
                     "f1_Retenido": f1_score(1 - yv, 1 - pred),
                     "f1_Deserción": f1_score(yv, pred),
                     "pr_auc": average_precision_score(yv, proba[:, 1])})

    Xt = make_preprocessor(num, cat).fit_transform(X)
    Xr, yr = SMOTE(random_state=seed).fit_resample(Xt, y)
    for k, (tr, va) in enumerate(cv.split(Xr, yr)):
        m = clone(clf).fit(Xr[tr], yr[tr])
        _record("Artículo (SMOTE antes de partir)", k, yr[va], m.predict(Xr[va]),
                m.predict_proba(Xr[va]))

    spec = ModelSpec("LightGBM", "", clf)
    for k, (tr, va) in enumerate(cv.split(X, y)):
        pipe = build_pipeline(spec, num, cat, sampler=SMOTE(random_state=seed)).fit(X.iloc[tr], y[tr])
        _record("Pipeline (SMOTE dentro del pliegue)", k, y[va], pipe.predict(X.iloc[va]),
                pipe.predict_proba(X.iloc[va]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 5. Isolation Forest
def isolation_forest(dev: pd.DataFrame) -> pd.DataFrame:
    seed = config.seed()
    num, cat = feature_lists()
    X, y, g = dev[num + cat], dev[TARGET], dev[GROUP]
    rows = []
    for stage, contamination in [("Etapa 1 (auto)", "auto"), ("Etapa 2 (0,5)", 0.5)]:
        for k, (tr, va) in enumerate(_outer(seed).split(X, y, g)):
            prep = make_preprocessor(num, cat).fit(X.iloc[tr])
            iso = IsolationForest(n_estimators=300, contamination=contamination,
                                  random_state=seed, n_jobs=-1).fit(prep.transform(X.iloc[tr]))
            Xv = prep.transform(X.iloc[va])
            flagged = iso.predict(Xv) == -1
            yv = y.iloc[va].to_numpy()
            rows.append({"etapa": stage, "pliegue": k,
                         "pct_desertores_anomalos": 100 * flagged[yv == 1].mean(),
                         "pct_retenidos_anomalos": 100 * flagged[yv == 0].mean(),
                         "roc_auc_puntaje": roc_auc_score(yv, -iso.score_samples(Xv)),
                         "pr_auc_puntaje": average_precision_score(yv, -iso.score_samples(Xv))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- 6. VIF
def vif(dev: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """VIF de las numéricas del modelo (``extra`` permite ver el efecto de una variable excluida)."""
    num, _ = feature_lists()
    cols = num + (extra or [])
    Xn = dev[cols].astype(float)
    Xn = Xn.loc[:, Xn.std() > 0]
    Xn = (Xn.fillna(Xn.median()) - Xn.mean()) / Xn.std()
    Xn = Xn.assign(const=1.0)
    out = [{"variable": c, "vif": variance_inflation_factor(Xn.values, i)}
           for i, c in enumerate(Xn.columns) if c != "const"]
    return pd.DataFrame(out).sort_values("vif", ascending=False)


# ------------------------------------------------------- 7. Curvas de validación
def validation_curve(dev: pd.DataFrame, spec: ModelSpec, params: dict, name: str,
                     values: list) -> pd.DataFrame:
    """PR-AUC de entrenamiento y validación al variar un hiperparámetro (sesgo y varianza).

    Los demás hiperparámetros quedan en los valores que eligió la validación cruzada anidada,
    de modo que la curva muestra la sensibilidad alrededor de la configuración seleccionada.
    """
    seed = config.seed()
    num, cat = feature_lists()
    X, y, g = dev[num + cat], dev[TARGET], dev[GROUP]
    rows = []
    for v in values:
        for k, (tr, va) in enumerate(_outer(seed).split(X, y, g)):
            pipe = tuned_pipeline(spec, {**params, name: v}, num, cat).fit(X.iloc[tr], y.iloc[tr])
            for part, idx in (("entrenamiento", tr), ("validación", va)):
                s = _scores(pipe, X.iloc[idx])
                rows.append({"modelo": spec.name, "parametro": name, "valor": v, "pliegue": k,
                             "parte": part, "pr_auc": average_precision_score(y.iloc[idx], s)})
        log.info("Curva de validación | %s %s=%s lista", spec.name, name, v)
    return pd.DataFrame(rows)


def _scores(pipe, X) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


# --------------------------------------------------- 8. Costo computacional
def computational_cost(dev: pd.DataFrame, folds: pd.DataFrame, specs: dict[str, ModelSpec],
                       repeats: int = 3) -> pd.DataFrame:
    """Costo de cada modelo con los hiperparámetros elegidos, sobre todo el conjunto de desarrollo.

    - Búsqueda: tiempo total de la validación cruzada anidada (rejilla + umbral, cinco rondas).
    - Ajuste: mediana de ``repeats`` entrenamientos del Pipeline completo (incluye SMOTE).
    - Predicción: tiempo por cada 1.000 estudiantes.
    - Tamaño: Pipeline serializado con pickle.
    """
    num, cat = feature_lists()
    X, y = dev[num + cat], dev[TARGET]
    rows = []
    for name, spec in specs.items():
        params = folds.loc[folds.modelo == name, "params"].mode().iloc[0]
        pipe_params = params_from_text(params)
        fit_t, pred_t = [], []
        for _ in range(repeats):
            pipe = tuned_pipeline(spec, pipe_params, num, cat)
            t0 = time.perf_counter()
            pipe.fit(X, y)
            fit_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _scores(pipe, X)
            pred_t.append(time.perf_counter() - t0)
        rows.append({"modelo": name,
                     "busqueda_min": folds.loc[folds.modelo == name, "segundos"].sum() / 60,
                     "ajuste_s": float(np.median(fit_t)),
                     "prediccion_ms_por_1000": 1000 * float(np.median(pred_t)) / len(X) * 1000,
                     "tamano_mb": len(pickle.dumps(pipe)) / 1e6,
                     "combinaciones": int(np.prod([len(v) for v in spec.grid.values()]))
                     if spec.grid else 1,
                     "nucleos": os.cpu_count(), "procesador": platform.processor()})
        log.info("Costo | %s: ajuste %.1fs", name, rows[-1]["ajuste_s"])
    return pd.DataFrame(rows)


# --------------------------------------------- 9. Sensibilidad del modelo final
FINAL_VARIANTS = ("Base (Segunda entrega)", "Sin marca de faltante de la rúbrica",
                  "Sin marcas de faltante", "Con la cohorte como variable")


def _final_variant(spec: ModelSpec, name: str) -> tuple[ModelSpec, list[str], list[str]]:
    """Variantes del modelo final para poner a prueba el papel de los datos faltantes.

    La rúbrica de admisión solo falta en AD19 (45 %) y nunca en AD20, por lo que su marca de
    faltante podría funcionar como indicador de cohorte. Se compara el modelo base con uno sin
    esa marca, uno sin ninguna marca y uno que recibe la cohorte de forma explícita.
    """
    num, cat = feature_lists()
    if name == "Sin marca de faltante de la rúbrica":
        return replace(spec, no_indicator=("admission.rubric",)), num, cat
    if name == "Sin marcas de faltante":
        return replace(spec, no_indicator=tuple(num)), num, cat
    if name == "Con la cohorte como variable":
        return spec, num, cat + ["generation"]
    return spec, num, cat


def missing_flag_sensitivity(dev: pd.DataFrame, spec: ModelSpec,
                             variants=FINAL_VARIANTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CV anidada (misma rejilla y protocolo que la comparación de modelos) de cada variante."""
    folds, oofs = [], []
    for name in variants:
        s, num, cat = _final_variant(spec, name)
        f, o = nested_cv(s, dev[num + cat], dev[TARGET], dev[GROUP], num, cat)
        folds.append(f.assign(variante=name))
        oofs.append(o.assign(variante=name))
        log.info("Sensibilidad | %s: PR-AUC %.3f", name, f["pr_auc"].mean())
    return pd.concat(folds, ignore_index=True), pd.concat(oofs, ignore_index=True)


def _capture(y: np.ndarray, s: np.ndarray, frac: float = 0.20) -> float:
    top = np.argsort(-s, kind="stable")[: int(round(frac * len(s)))]
    return float(y[top].sum() / y.sum())


def temporal_validation(dev: pd.DataFrame, spec: ModelSpec,
                        variants=FINAL_VARIANTS[:2]) -> pd.DataFrame:
    """Entrena con una cohorte y evalúa en la otra (AD19 -> AD20 es el uso real del modelo).

    Dentro de la cohorte de entrenamiento se repite el protocolo de la CV anidada: rejilla con
    CV interna agrupada y umbral F2 con predicciones fuera de pliegue. La cohorte evaluada
    nunca interviene en esas decisiones.
    """
    cfg, seed = config.load_config(), config.seed()
    beta = cfg["metrics"]["threshold_beta"]
    rows = []
    for gen_tr, gen_te in (("AD19", "AD20"), ("AD20", "AD19")):
        tr, te = dev[dev["generation"] == gen_tr], dev[dev["generation"] == gen_te]
        for name in variants:
            s, num, cat = _final_variant(spec, name)
            search = GridSearchCV(build_pipeline(s, num, cat), s.grid,
                                  scoring=cfg["metrics"]["tuning"],
                                  cv=_outer(seed + 1, cfg["cv"]["inner_folds"]),
                                  n_jobs=s.search_n_jobs, refit=True, error_score="raise")
            search.fit(tr[num + cat], tr[TARGET], groups=tr[GROUP])
            model = search.best_estimator_
            oof = cross_val_predict(clone(model), tr[num + cat], tr[TARGET], groups=tr[GROUP],
                                    cv=_outer(seed + 2, cfg["cv"]["threshold_folds"]),
                                    method="predict_proba")[:, 1]
            thr = best_threshold(tr[TARGET].to_numpy(), oof, beta)
            y, sc = te[TARGET].to_numpy(), model.predict_proba(te[num + cat])[:, 1]
            m = binary_metrics(y, sc, (sc >= thr).astype(int), beta)
            rows.append({"esquema": "Temporal", "entrena": gen_tr, "evalua": gen_te,
                         "variante": name, "n_entrena": len(tr), "n_evalua": len(te),
                         "prevalencia": y.mean(), **m, "pr_auc_sobre_azar": m["pr_auc"] / y.mean(),
                         "captura_top20": _capture(y, sc), "prob_media": sc.mean(),
                         "params": params_to_text(search.best_params_)})
            log.info("Temporal | %s -> %s | %s: PR-AUC %.3f", gen_tr, gen_te, name, m["pr_auc"])
    return pd.DataFrame(rows)


def cohort_reference(oof: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    """Las mismas métricas, por cohorte, de la CV anidada que mezcla AD19 y AD20 (referencia)."""
    oof = oof.assign(generation=dev.loc[oof["row_id"], "generation"].to_numpy())
    rows = []
    for (name, gen), d in oof.groupby(["variante", "generation"], sort=False):
        y, sc = d["y"].to_numpy(), d["score"].to_numpy()
        pr = average_precision_score(y, sc)
        rows.append({"esquema": "CV mezclando cohortes", "evalua": gen, "variante": name,
                     "n_evalua": len(d), "prevalencia": y.mean(), "pr_auc": pr,
                     "roc_auc": roc_auc_score(y, sc), "recall": recall_score(y, d["pred"]),
                     "pr_auc_sobre_azar": pr / y.mean(), "captura_top20": _capture(y, sc),
                     "prob_media": sc.mean()})
    return pd.DataFrame(rows)


def calibration(oof: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """¿Las probabilidades significan lo que dicen? Puntaje de Brier y curva de calibración.

    Por pliegue: Brier del modelo frente al de predecir siempre la tasa base, probabilidad media
    frente a tasa observada, y pendiente e intercepto de calibración (regresión logística de y
    sobre logit(p); lo ideal es pendiente 1 e intercepto 0). La curva agrupa las predicciones
    fuera de pliegue en deciles de probabilidad.
    """
    from desercion.eda import wilson

    rows = []
    for k, d in oof.groupby("pliegue"):
        y, p = d["y"].to_numpy(), np.clip(d["score"].to_numpy(), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(penalty=None).fit(logit, y)
        rows.append({"pliegue": k, "brier": brier_score_loss(y, p),
                     "brier_tasa_base": y.mean() * (1 - y.mean()), "prob_media": p.mean(),
                     "tasa_observada": y.mean(), "pendiente": lr.coef_[0, 0],
                     "intercepto": lr.intercept_[0]})
    per_fold = pd.DataFrame(rows)
    d = oof.assign(bin=pd.qcut(oof["score"], bins, labels=False, duplicates="drop"))
    curve = d.groupby("bin").agg(prob_media=("score", "mean"), tasa_observada=("y", "mean"),
                                 desertores=("y", "sum"), n=("y", "size")).reset_index()
    curve["ic_bajo"], curve["ic_alto"] = wilson(curve["desertores"], curve["n"])
    return per_fold, curve
