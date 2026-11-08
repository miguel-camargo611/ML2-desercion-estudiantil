"""Catálogo de modelos: líneas base, referencia del curso anterior y modelos avanzados del artículo.

Cada entrada define el estimador y la rejilla de hiperparámetros para ``GridSearchCV``
(sustituto de Optuna acordado en la Primera entrega). El balanceo de clases es un
hiperparámetro más: la validación cruzada interna decide si SMOTE (o, en CatBoost, los pesos
de clase) mejora a cada modelo.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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

NO_SAMPLER = "passthrough"


class CatBoostNative(ClassifierMixin, BaseEstimator):
    """CatBoost con categóricas nativas y una interfaz estándar de scikit-learn.

    Se envuelve en lugar de heredar porque CatBoost descarta en su constructor los parámetros
    con valor ``None``, lo que impide que ``GridSearchCV`` clone el modelo con la opción "sin
    pesos de clase". Aquí esa opción es explícita: ``class_weights="none"``.
    """

    def __init__(self, iterations: int = 400, depth: int = 6, learning_rate: float = 0.05,
                 l2_leaf_reg: float = 3.0, class_weights: str = "none",
                 random_state: int | None = None, thread_count: int = -1):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.class_weights = class_weights
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X, y):
        cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        weights = {} if self.class_weights == "none" else {"auto_class_weights": self.class_weights}
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, random_seed=self.random_state,
            thread_count=self.thread_count, verbose=0, allow_writing_files=False, **weights)
        self.model_.fit(X, y, cat_features=cats)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator: BaseEstimator
    grid: dict[str, list[Any]] = field(default_factory=dict)
    native_categoricals: bool = False
    search_n_jobs: int = 1
    # Numéricas que se imputan sin la columna indicadora de faltante (análisis de sensibilidad).
    no_indicator: tuple[str, ...] = ()


def _balance(seed: int) -> dict[str, list[Any]]:
    """Opción de balanceo que explora la rejilla: sin remuestreo o SMOTE."""
    return {"sampler": [NO_SAMPLER, SMOTE(random_state=seed)]}


def get_specs(seed: int | None = None) -> dict[str, ModelSpec]:
    seed = config.seed() if seed is None else seed
    bal = _balance(seed)
    specs = [
        ModelSpec("Mayoría", "Línea base", DummyClassifier(strategy="prior")),
        ModelSpec("Árbol sin poda", "Línea base (ML1)", DecisionTreeClassifier(random_state=seed)),
        ModelSpec(
            "Regresión logística", "Referencia (ML1)",
            LogisticRegression(max_iter=5000, random_state=seed),
            grid={"clf__C": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1], **bal}, search_n_jobs=-1,
        ),
        ModelSpec(
            "Random Forest", "Bagging",
            RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed),
            grid={"clf__min_samples_leaf": [5, 15, 40, 100], "clf__max_features": ["sqrt", 0.4],
                  **bal},
        ),
        ModelSpec(
            "SVM (RBF-Nystroem)", "SVM y kernels",
            Pipeline([
                ("kernel", Nystroem(kernel="rbf", n_components=500, random_state=seed)),
                ("svm", LinearSVC(dual=False, max_iter=5000, random_state=seed)),
            ]),
            grid={"clf__kernel__gamma": [0.001, 0.003, 0.01], "clf__svm__C": [0.001, 0.01, 0.1],
                  **bal},
            search_n_jobs=-1,
        ),
        ModelSpec(
            "Gradient Boosting", "Boosting",
            HistGradientBoostingClassifier(max_iter=400, early_stopping=False, random_state=seed),
            grid={"clf__learning_rate": [0.02, 0.05], "clf__max_leaf_nodes": [4, 8, 16],
                  "clf__l2_regularization": [0.0, 1.0], **bal},
            search_n_jobs=-1,
        ),
        ModelSpec(
            "XGBoost", "Boosting",
            XGBClassifier(n_estimators=400, tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", n_jobs=-1, random_state=seed),
            grid={"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.02, 0.05],
                  "clf__min_child_weight": [1, 10], **bal},
        ),
        ModelSpec(
            "LightGBM", "Boosting",
            LGBMClassifier(n_estimators=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           n_jobs=-1, random_state=seed, verbose=-1),
            grid={"clf__num_leaves": [4, 8, 16], "clf__learning_rate": [0.02, 0.05],
                  "clf__min_child_samples": [20, 100], **bal},
        ),
        ModelSpec(
            "CatBoost", "Boosting",
            CatBoostNative(iterations=400, random_state=seed),
            # SMOTE no admite categóricas sin codificar: el balanceo se explora con pesos de clase.
            # Rejilla reducida por costo: es el modelo más caro de ajustar (ver costo_computacional);
            # la tasa de aprendizaje y la regularización L2 quedan en 0,05 y 3 (valor por defecto).
            grid={"clf__depth": [4, 6], "clf__class_weights": ["none", "Balanced"]},
            native_categoricals=True,
        ),
    ]
    return {s.name: s for s in specs}


def build_pipeline(spec: ModelSpec, num: list[str], cat: list[str], seed: int | None = None,
                   sampler: Any = NO_SAMPLER) -> ImbPipeline:
    """Pipeline preprocesamiento -> balanceo -> clasificador.

    Se usa el Pipeline de imbalanced-learn: el remuestreo solo ocurre en ``fit`` (datos de
    entrenamiento del pliegue) y nunca en ``predict``/``transform`` sobre validación o prueba.
    El paso ``sampler`` existe siempre para que la rejilla pueda activarlo o no.
    """
    prep = (make_native_preprocessor(num, cat) if spec.native_categoricals
            else make_preprocessor(num, cat, spec.no_indicator))
    return ImbPipeline([("prep", prep), ("sampler", sampler), ("clf", clone(spec.estimator))])


# ------------------------------------------------- hiperparámetros como texto legible
def params_to_text(params: dict[str, Any]) -> str:
    """Serializa los hiperparámetros elegidos (los objetos de balanceo se guardan por nombre)."""
    plain = {k: (v if isinstance(v, (int, float, str, type(None))) else type(v).__name__)
             for k, v in params.items()}
    return str(plain)


def params_from_text(text: str, seed: int | None = None) -> dict[str, Any]:
    """Inverso de ``params_to_text``: reconstruye los objetos de balanceo."""
    seed = config.seed() if seed is None else seed
    params = ast.literal_eval(text)
    if params.get("sampler") == "SMOTE":
        params["sampler"] = SMOTE(random_state=seed)
    return params


def tuned_pipeline(spec: ModelSpec, params: dict[str, Any], num: list[str],
                   cat: list[str]) -> ImbPipeline:
    """Pipeline con los hiperparámetros elegidos ya fijados."""
    return build_pipeline(spec, num, cat).set_params(**params)
