"""Experimentos que reproducen (y ponen a prueba) decisiones concretas del artículo base.

1. ``resampling_comparison``  – SMOTE vs ADASYN con un árbol de decisión de control (Tabla 6).
2. ``leakage_demo``           – protocolo del artículo (SMOTE antes de partir) vs Pipeline correcto.
3. ``isolation_forest``       – ¿son los desertores "anomalías"? (etapas 1 y 2 del artículo).
4. ``ablation``               – pregunta de investigación: socioeconómicas vs académicas.
5. ``multiclass``             – formulación de tres clases análoga a Graduate/Enrolled/Dropout.
6. ``vif``                    – ¿se replica la multicolinealidad que llevó a descartar la RL?
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

from desercion import config
from desercion.data import GROUP, MULTI_LABELS, TARGET, TARGET_MULTI
from desercion.evaluate import best_threshold, binary_metrics
from desercion.features import GROUPS, feature_lists, make_preprocessor
from desercion.models import ModelSpec, build_pipeline, get_specs

log = logging.getLogger(__name__)

# Configuración fija de LightGBM para experimentos (la más elegida en la CV anidada preliminar).
LGBM_FIXED = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def _outer(seed: int, n: int = 5) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)


def _lgbm_spec(seed: int, **extra) -> ModelSpec:
    return ModelSpec("LightGBM", "Boosting",
                     LGBMClassifier(random_state=seed, n_jobs=-1, **{**LGBM_FIXED, **extra}))


# -------------------------------------------------------------- 1. SMOTE vs ADASYN
def resampling_comparison(dev: pd.DataFrame, scenario: str) -> pd.DataFrame:
    seed = config.seed()
    num, cat = feature_lists(scenario)
    X, y, g = dev[num + cat], dev[TARGET], dev[GROUP]
    control = ModelSpec("Árbol de control", "", DecisionTreeClassifier(random_state=seed))
    variants = {
        "Sin balanceo": (control, None),
        "Pesos de clase": (ModelSpec("Árbol", "", DecisionTreeClassifier(
            random_state=seed, class_weight="balanced")), None),
        "SMOTE": (control, SMOTE(random_state=seed)),
        "ADASYN": (control, ADASYN(random_state=seed)),
    }
    rows = []
    for name, (spec, sampler) in variants.items():
        for k, (tr, va) in enumerate(_outer(seed).split(X, y, g)):
            pipe = build_pipeline(spec, num, cat, sampler=sampler).fit(X.iloc[tr], y.iloc[tr])
            yv, pred = y.iloc[va].to_numpy(), pipe.predict(X.iloc[va])
            rows.append({"tecnica": name, "pliegue": k,
                         "f1_retenido": f1_score(1 - yv, 1 - pred),
                         "f1_desercion": f1_score(yv, pred),
                         "f1_ponderado": f1_score(yv, pred, average="weighted"),
                         "recall_desercion": recall_score(yv, pred)})
        log.info("Balanceo %s listo", name)
    return pd.DataFrame(rows)


# -------------------------------------------------- 2. Demostración de fuga (SMOTE)
def leakage_demo(dev: pd.DataFrame, scenario: str, multiclass: bool = False) -> pd.DataFrame:
    """Compara el protocolo del artículo con el Pipeline sin fuga, mismo modelo e hiperparámetros.

    - "Artículo (SMOTE antes de partir)": se preprocesa y sobremuestrea TODO el conjunto y luego
      se hace la CV: los pliegues de validación contienen puntos sintéticos interpolados a partir
      de vecinos que están en entrenamiento.
    - "Pipeline (SMOTE dentro del pliegue)": el remuestreo solo ve el entrenamiento de cada pliegue.
    """
    seed = config.seed()
    num, cat = feature_lists(scenario)
    target = TARGET_MULTI if multiclass else TARGET
    X, y = dev[num + cat], dev[target].to_numpy()
    labels = sorted(np.unique(y))
    names = [MULTI_LABELS[c] for c in labels] if multiclass else ["Retenido", "Deserción"]
    clf = LGBMClassifier(random_state=seed, n_jobs=-1, **LGBM_FIXED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []

    def _record(protocol, k, yv, pred):
        f1s = f1_score(yv, pred, labels=labels, average=None)
        row = {"protocolo": protocol, "pliegue": k,
               "f1_ponderado": f1_score(yv, pred, average="weighted"),
               "f1_macro": f1_score(yv, pred, average="macro")}
        row.update({f"f1_{n}": v for n, v in zip(names, f1s, strict=True)})
        rows.append(row)

    # Protocolo del artículo
    Xt = make_preprocessor(num, cat).fit_transform(X)
    Xr, yr = SMOTE(random_state=seed).fit_resample(Xt, y)
    for k, (tr, va) in enumerate(cv.split(Xr, yr)):
        m = clone(clf).fit(Xr[tr], yr[tr])
        _record("Artículo (SMOTE antes de partir)", k, yr[va], m.predict(Xr[va]))

    # Pipeline correcto
    spec = ModelSpec("LightGBM", "", clf)
    for k, (tr, va) in enumerate(cv.split(X, y)):
        pipe = build_pipeline(spec, num, cat).fit(X.iloc[tr], y[tr])
        _record("Pipeline (SMOTE dentro del pliegue)", k, y[va], pipe.predict(X.iloc[va]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 3. Isolation Forest
def isolation_forest(dev: pd.DataFrame, scenario: str) -> pd.DataFrame:
    seed = config.seed()
    num, cat = feature_lists(scenario)
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


# ------------------------------------------------------------------ 4. Ablación
ABLATION_SETS = {
    "A": {
        "Socioeconómicas": ["socioeconomicas"],
        "Demográficas + socioeconómicas": ["demograficas", "socioeconomicas"],
        "Académicas de ingreso": ["academicas_ingreso"],
        "Institucionales": ["institucionales"],
        "Todo al ingreso": ["demograficas", "socioeconomicas", "academicas_ingreso",
                            "institucionales"],
    },
    "B": {
        "Demográficas + socioeconómicas": ["demograficas", "socioeconomicas"],
        "Todo al ingreso": ["demograficas", "socioeconomicas", "academicas_ingreso",
                            "institucionales"],
        "Solo primer periodo": ["primer_periodo"],
        "Ingreso + primer periodo": list(GROUPS),
    },
}


def ablation(dev: pd.DataFrame, scenario: str) -> pd.DataFrame:
    seed = config.seed()
    beta = config.load_config()["metrics"]["threshold_beta"]
    y, g = dev[TARGET], dev[GROUP]
    rows = []
    for set_name, groups in ABLATION_SETS[scenario].items():
        num, cat = feature_lists(scenario, groups)
        X = dev[num + cat]
        for k, (tr, va) in enumerate(_outer(seed).split(X, y, g)):
            pipe = build_pipeline(_lgbm_spec(seed), num, cat)
            oof = cross_val_predict(clone(pipe), X.iloc[tr], y.iloc[tr], groups=g.iloc[tr],
                                    cv=_outer(seed + 2, 3), method="predict_proba")[:, 1]
            thr = best_threshold(y.iloc[tr].to_numpy(), oof, beta)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            s = pipe.predict_proba(X.iloc[va])[:, 1]
            m = binary_metrics(y.iloc[va].to_numpy(), s, (s >= thr).astype(int), beta)
            rows.append({"conjunto": set_name, "pliegue": k, "n_variables": len(num) + len(cat), **m})
        log.info("Ablación %s | %s listo", scenario, set_name)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. Multiclase
def multiclass(dev: pd.DataFrame, scenario: str) -> pd.DataFrame:
    seed = config.seed()
    num, cat = feature_lists(scenario)
    X, y, g = dev[num + cat], dev[TARGET_MULTI], dev[GROUP]
    specs = get_specs(seed)
    chosen = {
        "Mayoría": specs["Mayoría"],
        "Árbol sin poda": specs["Árbol sin poda"],
        "LightGBM": _lgbm_spec(seed),
        "CatBoost": ModelSpec("CatBoost", "", clone(specs["CatBoost"].estimator).set_params(
            depth=6, learning_rate=0.1)),
    }
    rows = []
    for name, spec in chosen.items():
        for k, (tr, va) in enumerate(_outer(seed).split(X, y, g)):
            pipe = build_pipeline(spec, num, cat).fit(X.iloc[tr], y.iloc[tr])
            yv, pred = y.iloc[va].to_numpy(), np.asarray(pipe.predict(X.iloc[va])).ravel().astype(int)
            f1s = f1_score(yv, pred, labels=[0, 1, 2], average=None, zero_division=0)
            rec = recall_score(yv, pred, labels=[0, 1, 2], average=None, zero_division=0)
            rows.append({"modelo": name, "pliegue": k,
                         **{f"f1_{MULTI_LABELS[c]}": f1s[c] for c in range(3)},
                         **{f"recall_{MULTI_LABELS[c]}": rec[c] for c in range(3)},
                         "f1_ponderado": f1_score(yv, pred, average="weighted", zero_division=0),
                         "f1_macro": f1_score(yv, pred, average="macro", zero_division=0)})
        log.info("Multiclase %s listo", name)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- 6. VIF
def vif(dev: pd.DataFrame, scenario: str) -> pd.DataFrame:
    num, _ = feature_lists(scenario)
    Xn = dev[num].astype(float)
    Xn = Xn.loc[:, Xn.std() > 0]
    Xn = (Xn.fillna(Xn.median()) - Xn.mean()) / Xn.std()
    Xn = Xn.assign(const=1.0)
    out = [{"variable": c, "vif": variance_inflation_factor(Xn.values, i)}
           for i, c in enumerate(Xn.columns) if c != "const"]
    return pd.DataFrame(out).sort_values("vif", ascending=False)
