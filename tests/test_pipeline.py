import numpy as np
import pytest
from imblearn.over_sampling import SMOTE

from desercion import data
from desercion.evaluate import best_threshold, compare_models
from desercion.features import feature_lists, make_preprocessor
from desercion.models import build_pipeline, get_specs


@pytest.fixture(scope="module")
def xy(raw_like):
    df = data.add_targets(data.clean(raw_like))
    num, cat = feature_lists("B")
    return df[num + cat], df[data.TARGET], num, cat


@pytest.mark.parametrize("name", ["Mayoría", "Árbol sin poda", "LightGBM", "CatBoost nativo",
                                  "SVM (RBF-Nystroem)"])
def test_pipelines_fit_and_score(xy, name):
    X, y, num, cat = xy
    spec = get_specs()[name]
    pipe = build_pipeline(spec, num, cat).fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == len(X)


def test_smote_only_resamples_during_fit(xy):
    """El Pipeline de imblearn no debe cambiar el número de filas al predecir."""
    X, y, num, cat = xy
    pipe = build_pipeline(get_specs()["LightGBM"], num, cat).fit(X, y)
    assert isinstance(pipe.named_steps["sampler"], SMOTE)
    assert pipe.predict_proba(X).shape[0] == len(X)


def test_preprocessor_learns_only_from_fit_data(xy):
    X, _, num, cat = xy
    half = len(X) // 2
    prep = make_preprocessor(num, cat).fit(X.iloc[:half])
    medians = prep.named_transformers_["num"].named_steps["imputar"].statistics_
    assert np.allclose(medians, X.iloc[:half][num].astype(float).median().to_numpy(), equal_nan=True)


def test_best_threshold_prefers_recall():
    y = np.array([0] * 90 + [1] * 10)
    s = np.r_[np.linspace(0, 0.6, 90), np.linspace(0.4, 1, 10)]
    recall_f2 = ((s >= best_threshold(y, s, beta=2))[y == 1]).mean()
    recall_f1 = ((s >= best_threshold(y, s, beta=1))[y == 1]).mean()
    assert recall_f2 >= recall_f1


def test_compare_models_detects_identical():
    import pandas as pd

    folds = pd.DataFrame({"modelo": ["a"] * 5 + ["b"] * 5, "pliegue": list(range(5)) * 2,
                          "pr_auc": [0.5, 0.52, 0.51, 0.49, 0.5] * 2})
    r = compare_models(folds, "a", "b", "pr_auc", n_train=800, n_test=200)
    assert r["diferencia"] == 0


def test_original_variable_mapping():
    from desercion.interpret import original_variable

    originals = ["school", "school.cost", "PNA", "program"]
    assert original_variable("school.cost_High cost", originals) == "school.cost"
    assert original_variable("school_EIC", originals) == "school"
    assert original_variable("missingindicator_PNA", originals) == "PNA"
    assert original_variable("program_infrequent_sklearn", originals) == "program"
