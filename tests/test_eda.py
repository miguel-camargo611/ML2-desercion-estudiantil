import numpy as np
import pandas as pd

from desercion import data, eda
from desercion.features import add_engineered, feature_lists


def _clean(raw_like):
    return add_engineered(data.add_targets(data.clean(raw_like)))


def test_wilson_interval_contains_proportion():
    lo, hi = eda.wilson(np.array([5, 0, 50]), np.array([100, 40, 50]))
    p = np.array([0.05, 0.0, 1.0])
    assert np.all(lo <= p + 1e-12) and np.all(p <= hi + 1e-12)
    assert np.all((lo >= 0) & (hi <= 1))


def test_target_by_cohort_adds_up(raw_like):
    df = _clean(raw_like)
    t = eda.target_by_cohort(df)
    expected = df.groupby("generation")[data.TARGET].mean()
    assert np.allclose(t["tasa_total"], expected.loc[t.index])


def test_missing_counts_no_information_and_nan(raw_like):
    df = _clean(raw_like)
    miss = eda.missing_by_cohort(df, ["socioeconomic.level", "general.math.eval"])
    s = df["socioeconomic.level"]
    by_hand = (s.eq("No information") | s.isna()).groupby(df["generation"]).mean() * 100
    assert np.allclose(miss.loc["socioeconomic.level"], by_hand.round(1).loc[miss.columns])
    assert (miss.loc["general.math.eval"] > 0).any()  # "No information" numérico -> NaN


def test_rate_tables_are_consistent(raw_like):
    df = _clean(raw_like)
    cat = eda.rate_by_category(df, "region", min_n=1)
    assert cat["n"].sum() == len(df)
    assert ((cat["ic_inf"] <= cat["tasa"]) & (cat["tasa"] <= cat["ic_sup"])).all()
    num = eda.rate_by_quantile(df, "average.first.period", q=5)
    assert num["n"].sum() == len(df)


def test_univariate_association_detects_signal(raw_like):
    df = _clean(raw_like)
    num, cat = feature_lists()
    t = eda.univariate_association(df, num, cat)
    assert set(t["variable"]) == set(num + cat)
    # En los datos sintéticos la deserción depende del promedio del primer periodo.
    assert t.iloc[0]["variable"] == "average.first.period"
    assert t.set_index("variable").loc["average.first.period", "auc"] < 0.5


def test_cramers_v_bounds():
    x = pd.Series(list("ab" * 50))
    assert eda.cramers_v(x, x) > 0.9
    assert eda.cramers_v(x, pd.Series([0] * 100)) == 0.0
