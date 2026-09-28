"""Catálogo de modelos: línea base, referencia ML1 y modelos avanzados del artículo.

Cada entrada define el estimador, si usa SMOTE dentro del Pipeline y la rejilla de
hiperparámetros para ``GridSearchCV`` (sustituto de Optuna acordado en la Primera entrega).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from desercion import config
from desercion.features import make_native_preprocessor, make_preprocessor


class CatBoostNative(CatBoostClassifier):
    """CatBoost que detecta las columnas categóricas en ``fit`` (clonable por scikit-learn)."""

    def fit(self, X, y=None, **kwargs):
        cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        return super().fit(X, y, cat_features=cats, **kwargs)


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator: BaseEstimator
    grid: dict[str, list[Any]] = field(default_factory=dict)
    smote: bool = True
    native_categoricals: bool = False
    search_n_jobs: int = 1


def _n_jobs() -> int:
    return -1


def get_specs(seed: int | None = None) -> dict[str, ModelSpec]:
    seed = config.seed() if seed is None else seed
    specs = [
        ModelSpec("Mayoría", "Trivial", DummyClassifier(strategy="prior"), smote=False),
        ModelSpec("Árbol sin poda", "Línea base (ML1)", DecisionTreeClassifier(random_state=seed)),
        ModelSpec(
            "Regresión logística", "Referencia (ML1)",
            LogisticRegression(max_iter=3000, random_state=seed),
            grid={"clf__C": [0.0003, 0.001, 0.003, 0.01, 0.03]}, search_n_jobs=5,
        ),
        ModelSpec(
            "Random Forest", "Bagging",
            RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=_n_jobs(),
                                   random_state=seed),
            grid={"clf__max_depth": [None], "clf__min_samples_leaf": [10, 30, 100]},
        ),
        ModelSpec(
            "SVM (RBF-Nystroem)", "SVM y kernels",
            Pipeline([
                ("kernel", Nystroem(kernel="rbf", n_components=400, random_state=seed)),
                ("svm", LinearSVC(dual=False, max_iter=5000, random_state=seed)),
            ]),
            grid={"clf__kernel__gamma": [0.001, 0.005], "clf__svm__C": [0.01, 0.1]},
            search_n_jobs=4,
        ),
        ModelSpec(
            "Gradient Boosting", "Boosting",
            HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=seed),
            grid={"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]},
        ),
        ModelSpec(
            "XGBoost", "Boosting",
            XGBClassifier(n_estimators=300, tree_method="hist", subsample=0.8,
                          colsample_bytree=0.8, eval_metric="logloss", n_jobs=_n_jobs(),
                          random_state=seed),
            grid={"clf__max_depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
        ),
        ModelSpec(
            "LightGBM", "Boosting",
            LGBMClassifier(n_estimators=300, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, n_jobs=_n_jobs(), random_state=seed, verbose=-1),
            grid={"clf__num_leaves": [4, 8, 15, 31], "clf__learning_rate": [0.02, 0.05]},
        ),
        ModelSpec(
            "CatBoost", "Boosting",
            CatBoostClassifier(iterations=300, verbose=0, random_seed=seed,
                               thread_count=_n_jobs(), allow_writing_files=False),
            grid={"clf__depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
        ),
        ModelSpec(
            "CatBoost nativo", "Boosting (adaptación)",
            CatBoostNative(iterations=300, verbose=0, random_seed=seed,
                           auto_class_weights="Balanced", thread_count=_n_jobs(),
                               allow_writing_files=False),
            grid={"clf__depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
            smote=False, native_categoricals=True,
        ),
    ]
    return {s.name: s for s in specs}


def build_pipeline(spec: ModelSpec, num: list[str], cat: list[str], seed: int | None = None,
                   sampler: Any | None = "default") -> ImbPipeline:
    """Pipeline preprocesamiento -> (SMOTE) -> clasificador.

    Se usa el Pipeline de imbalanced-learn: el remuestreo solo ocurre en ``fit`` (datos de
    entrenamiento del pliegue) y nunca en ``predict``/``transform`` sobre validación o prueba.
    """
    seed = config.seed() if seed is None else seed
    est = clone(spec.estimator)
    if spec.native_categoricals:
        prep = make_native_preprocessor(num, cat)
    else:
        prep = make_preprocessor(num, cat)
    steps: list[tuple[str, Any]] = [("prep", prep)]
    if sampler == "default":
        sampler = SMOTE(random_state=seed) if spec.smote else None
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", est))
    return ImbPipeline(steps)
