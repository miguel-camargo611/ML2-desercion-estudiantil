import numpy as np

from desercion import data
from desercion.features import (
    CANDIDATES,
    DROPPED_BY_SELECTION,
    EXCLUDED,
    KEPT_ENGINEERED,
    add_engineered,
    feature_lists,
)


def test_clean_converts_mixed_columns(raw_like):
    df = data.clean(raw_like)
    assert df["admission.test"].dtype.kind == "f"
    assert df["general.math.eval"].dtype.kind == "f"
    sin_examen = raw_like.index % 7 == 0
    en_linea = raw_like.index % 13 == 0
    assert df["admission.test"].isna().sum() == (sin_examen | en_linea).sum()
    assert not (df["PNA"] == 0).any()
    assert not (df["english.evaluation"] == 0).any()


def test_admission_scales_are_separated(raw_like):
    """El examen en línea (0-100) no debe mezclarse con el presencial (400-1600)."""
    df = data.clean(raw_like)
    assert (df["admission.test"].dropna() > 100).all()
    assert (df["admission.test.online"].dropna() <= 100).all()
    assert df["admission.test.online"].notna().sum() == (raw_like.index % 13 == 0).sum()


def test_target_is_binary_and_consistent(raw_like):
    df = data.add_targets(data.clean(raw_like))
    assert set(df[data.TARGET]) <= {0, 1}
    assert (df[data.TARGET] == (df["retention"] == 0)).all()


def test_no_leaky_or_unusable_columns_in_features():
    num, cat = feature_lists()
    used = set(num + cat)
    assert not {"retention", "dropout.semester", "student.id", "generation"} & used
    assert not set(EXCLUDED) & used
    assert not set(DROPPED_BY_SELECTION) & used
    assert {"socioeconomic.level", "social.lag"}.isdisjoint(CANDIDATES["num"] + CANDIDATES["cat"])


def test_engineered_features_are_row_wise(raw_like):
    df = add_engineered(data.clean(raw_like))
    gap = df["average.first.period"] - df["PNA"]
    assert np.allclose(df["brecha.promedio"], gap, equal_nan=True)
    num, _ = feature_lists()
    assert set(KEPT_ENGINEERED) <= set(num)


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


def test_study_frame_population(raw_like):
    df = data.add_targets(data.clean(raw_like))
    split = data.make_split(df, test_size=0.2, seed=1)
    dev = data.study_frame(df, split, "dev")
    assert (dev["dropout.semester"] != 1).all()
    assert set(dev["generation"]) <= {"AD19", "AD20"}
    assert (split.loc[dev.index] == "dev").all()


def test_raw_file_is_found_with_any_name(tmp_path, monkeypatch):
    import pytest

    from desercion import config

    monkeypatch.setattr(config, "path", lambda key: tmp_path)
    with pytest.raises(FileNotFoundError, match="data/raw"):
        data.find_raw_file()
    (tmp_path / "cualquier nombre.xlsx").touch()
    assert data.find_raw_file().name == "cualquier nombre.xlsx"
    (tmp_path / "otro.xlsx").touch()
    with pytest.raises(FileNotFoundError, match="2 archivos"):
        data.find_raw_file()
