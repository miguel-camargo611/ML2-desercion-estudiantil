"""Pruebas de regresión del contrato del modelo desplegado (como en el laboratorio de clase)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "app"))

import pandas as pd  # noqa: E402
from model_service import (  # noqa: E402
    example_batch,
    load_bundle,
    local_shap_values,
    make_frame,
    predict,
    predict_batch,
    risk_tier,
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(PROJECT / "artifacts")


@pytest.fixture(scope="module")
def reference():
    return json.loads((PROJECT / "artifacts" / "reference_prediction.json").read_text(
        encoding="utf-8"))


def test_feature_contract_matches_model(bundle) -> None:
    model, metadata = bundle
    assert list(model.feature_names_in_) == metadata["feature_order"]
    assert set(metadata["input_features"]) | set(metadata["engineered"]) == set(
        metadata["feature_order"])


def test_reference_prediction_is_stable(bundle, reference) -> None:
    """Paridad: el modelo cargado en este entorno reproduce la predicción del entrenamiento."""
    model, metadata = bundle
    result = predict(model, metadata, reference["input"])
    assert result["probability"] == pytest.approx(reference["probability"],
                                                  abs=reference["tolerance"])
    assert result["prediction"] == reference["prediction"]


def test_engineered_feature_uses_training_formula(bundle, reference) -> None:
    _, metadata = bundle
    row = make_frame(reference["input"], metadata).iloc[0]
    assert row["brecha.promedio"] == pytest.approx(row["average.first.period"] - row["PNA"])


def test_missing_feature_is_rejected(bundle) -> None:
    _, metadata = bundle
    with pytest.raises(ValueError, match="Faltan variables"):
        make_frame({}, metadata)


def test_unknown_category_and_out_of_range_are_rejected(bundle, reference) -> None:
    _, metadata = bundle
    with pytest.raises(ValueError, match="categoría desconocida"):
        make_frame({**reference["input"], "program": "NO-EXISTE"}, metadata)
    with pytest.raises(ValueError, match="fuera del rango"):
        make_frame({**reference["input"], "average.first.period": 150.0}, metadata)


def test_optional_inputs_accept_missing(bundle, reference) -> None:
    model, metadata = bundle
    optional = [f for f, s in metadata["input_contract"].items() if s.get("allow_missing")]
    assert "admission.test" in optional
    result = predict(model, metadata, {**reference["input"], **dict.fromkeys(optional)})
    assert 0 <= result["probability"] <= 1


def test_local_explanation_covers_every_variable(bundle, reference) -> None:
    model, metadata = bundle
    local = local_shap_values(model, metadata, reference["input"])
    assert set(local["variable"]) == set(metadata["feature_order"])


def test_batch_matches_single_predictions(bundle) -> None:
    """La carga masiva da exactamente lo mismo que predecir estudiante por estudiante."""
    model, metadata = bundle
    batch = example_batch(metadata, n=15)
    out = predict_batch(model, metadata, batch)
    assert (out["error"] == "").all()
    for i, row in batch.iterrows():
        single = predict(model, metadata, row.drop("id_estudiante").to_dict())
        assert out.loc[i, "probabilidad"] == pytest.approx(single["probability"], abs=1e-12)
        assert out.loc[i, "percentil_riesgo"] == single["percentile"]


def test_batch_reports_bad_rows_and_keeps_good_ones(bundle) -> None:
    model, metadata = bundle
    batch = example_batch(metadata, n=4)
    batch.loc[1, "program"] = "NO-EXISTE"
    batch["average.first.period"] = batch["average.first.period"].astype(object)  # como un CSV
    batch.loc[2, "average.first.period"] = "noventa"
    batch.loc[3, "admission.test"] = None  # admite dato faltante
    out = predict_batch(model, metadata, batch)
    assert "categoría desconocida" in out.loc[1, "error"]
    assert "no es un número" in out.loc[2, "error"]
    assert out.loc[[0, 3], "error"].eq("").all()
    assert out.loc[[0, 3], "probabilidad"].notna().all()
    assert out.loc[[1, 2], "probabilidad"].isna().all()


def test_batch_rejects_file_without_required_columns(bundle) -> None:
    model, metadata = bundle
    with pytest.raises(ValueError, match="Faltan variables"):
        predict_batch(model, metadata, pd.DataFrame({"id_estudiante": ["X"]}))


def test_risk_tiers_follow_threshold_and_top20(bundle) -> None:
    """Alto = 20 % de mayor riesgo; Medio = supera el umbral; Bajo = por debajo del umbral."""
    _, metadata = bundle
    thr, p80 = metadata["decision_threshold"], metadata["risk_percentiles"][79]
    assert thr < p80
    assert risk_tier(p80, metadata) == "Alto"
    assert risk_tier((thr + p80) / 2, metadata) == "Medio"
    assert risk_tier(thr / 2, metadata) == "Bajo"


def test_streamlit_config_is_the_same_for_docker_and_community_cloud() -> None:
    """Community Cloud lee .streamlit/ de la raíz; Docker, el de deploy/. Deben coincidir."""
    root = PROJECT.parent / ".streamlit" / "config.toml"
    local = PROJECT / ".streamlit" / "config.toml"
    assert root.read_text(encoding="utf-8") == local.read_text(encoding="utf-8")
