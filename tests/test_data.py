import numpy as np

from desercion import data
from desercion.features import EXCLUDED, SCENARIO_GROUPS, feature_lists


def test_clean_converts_mixed_columns(raw_like):
    df = data.clean(raw_like)
    assert df["admission.test"].dtype.kind == "f"
    assert df["general.math.eval"].dtype.kind == "f"
    assert df["admission.test"].isna().sum() == (raw_like.index % 7 == 0).sum()
    assert not (df["PNA"] == 0).any()
    assert not (df["english.evaluation"] == 0).any()


def test_targets_consistent(raw_like):
    df = data.add_targets(data.clean(raw_like))
    assert set(df[data.TARGET]) <= {0, 1}
    assert (df[data.TARGET] == (df["retention"] == 0)).all()
    # la versión multiclase es 0 si y solo si el estudiante fue retenido
    assert ((df[data.TARGET_MULTI] == 0) == (df[data.TARGET] == 0)).all()


def test_no_leaky_columns_in_features():
    leaky = {"retention", "dropout.semester", "student.id", "generation"}
    for scenario in SCENARIO_GROUPS:
        num, cat = feature_lists(scenario)
        assert not leaky & set(num + cat)
        assert not set(EXCLUDED) & set(num + cat)


def test_first_period_only_in_scenario_b():
    num_a, _ = feature_lists("A")
    num_b, _ = feature_lists("B")
    assert "average.first.period" not in num_a
    assert "average.first.period" in num_b


def test_split_is_grouped_and_stratified(raw_like):
    df = data.add_targets(data.clean(raw_like))
    split = data.make_split(df, test_size=0.2, seed=1)
    dev_ids = set(df.loc[split == "dev", data.GROUP])
    test_ids = set(df.loc[split == "test", data.GROUP])
    assert not dev_ids & test_ids
    frac = (split == "test").mean()
    assert 0.1 < frac < 0.3
    rate_dev = df.loc[split == "dev", data.TARGET].mean()
    rate_test = df.loc[split == "test", data.TARGET].mean()
    assert np.isclose(rate_dev, rate_test, atol=0.06)


def test_scenario_b_excludes_first_semester_dropouts(raw_like):
    df = data.add_targets(data.clean(raw_like))
    split = data.make_split(df, test_size=0.2, seed=1)
    b = data.scenario_frame(df, "B", split, "dev")
    assert (b["dropout.semester"] != 1).all()
    assert set(b["generation"]) <= {"AD19", "AD20"}
