import numpy as np
import pytest
from imblearn.over_sampling import SMOTE

from desercion import data
from desercion.evaluate import best_threshold, compare_models
from desercion.features import add_engineered, feature_lists, make_preprocessor
from desercion.models import (
    build_pipeline,
    get_specs,
    params_from_text,
    params_to_text,
    tuned_pipeline,
)


@pytest.fixture(scope="module")
def xy(raw_like):
    df = add_engineered(data.add_targets(data.clean(raw_like)))
    num, cat = feature_lists()
    return df[num + cat], df[data.TARGET], num, cat


@pytest.mark.parametrize("name", list(get_specs()))
def test_every_model_fits_with_its_first_grid_point(xy, name):
    X, y, num, cat = xy
    spec = get_specs()[name]
    first = {k: v[0] for k, v in spec.grid.items()}
    pipe = build_pipeline(spec, num, cat).set_params(**first).fit(X, y)
    assert len(pipe.predict(X)) == len(X)


def test_smote_only_resamples_during_fit(xy):
    """El Pipeline de imblearn no debe cambiar el número de filas al predecir."""
    X, y, num, cat = xy
    spec = get_specs()["LightGBM"]
    pipe = build_pipeline(spec, num, cat, sampler=SMOTE(random_state=0)).fit(X, y)
    assert isinstance(pipe.named_steps["sampler"], SMOTE)
    assert pipe.predict_proba(X).shape[0] == len(X)


def test_params_round_trip(xy):
    X, y, num, cat = xy
    params = {"clf__num_leaves": 8, "sampler": SMOTE(random_state=0)}
    back = params_from_text(params_to_text(params))
    assert back["clf__num_leaves"] == 8 and isinstance(back["sampler"], SMOTE)
    pipe = tuned_pipeline(get_specs()["LightGBM"], back, num, cat).fit(X, y)
    assert pipe.named_steps["clf"].num_leaves == 8


def test_preprocessor_learns_only_from_fit_data(xy):
    X, _, num, cat = xy
    half = len(X) // 2
    prep = make_preprocessor(num, cat).fit(X.iloc[:half])
    medians = prep.named_transformers_["num"].named_steps["imputar"].statistics_
    assert np.allclose(medians, X.iloc[:half][num].astype(float).median().to_numpy(), equal_nan=True)


def test_no_indicator_drops_only_the_requested_missing_flag(xy):
    """Sensibilidad: quitar la marca de faltante de la rúbrica no debe tocar las demás."""
    X, _, num, cat = xy
    base = make_preprocessor(num, cat).fit(X).get_feature_names_out()
    sin = make_preprocessor(num, cat, ("admission.rubric",)).fit(X).get_feature_names_out()
    assert "missingindicator_admission.rubric" in base
    assert "missingindicator_admission.rubric" not in sin
    assert set(base) - set(sin) == {"missingindicator_admission.rubric"}


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


def test_scaler_learns_only_from_training_rows(xy):
    """El escalamiento se ajusta con el entrenamiento del pliegue y se aplica tal cual a validación."""
    X, y, num, cat = xy
    half = len(X) // 2
    pipe = build_pipeline(get_specs()["Regresión logística"], num, cat).fit(X.iloc[:half],
                                                                          y.iloc[:half])
    num_pipe = pipe.named_steps["prep"].named_transformers_["num"]
    imputed = num_pipe.named_steps["imputar"].transform(X.iloc[:half][num])
    scaler = num_pipe.named_steps["escalar"]
    assert np.allclose(scaler.mean_, imputed.mean(axis=0))
    # Los datos no vistos se transforman con la media del entrenamiento, no con la suya propia.
    other = num_pipe.transform(X.iloc[half:][num])
    assert not np.allclose(other.mean(axis=0), 0)


def test_cost_and_validation_curve_run(raw_like):
    import pandas as pd

    from desercion import experiments
    from desercion.models import params_to_text

    df = add_engineered(data.add_targets(data.clean(raw_like)))
    specs = {k: v for k, v in get_specs().items() if k in ("Mayoría", "LightGBM")}
    folds = pd.DataFrame({"modelo": ["Mayoría", "LightGBM"], "segundos": [1.0, 2.0],
                          "params": ["{}", params_to_text({"clf__num_leaves": 4,
                                                           "sampler": "passthrough"})]})
    cost = experiments.computational_cost(df, folds, specs, repeats=1)
    assert set(cost.modelo) == set(specs) and (cost.ajuste_s > 0).all()
    curve = experiments.validation_curve(df, specs["LightGBM"], {"clf__num_leaves": 4},
                                         "clf__num_leaves", [2, 4])
    assert set(curve.parte) == {"entrenamiento", "validación"} and len(curve) == 2 * 5 * 2
