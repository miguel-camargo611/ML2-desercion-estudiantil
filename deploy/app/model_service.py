"""Servicios de la app: cargar el modelo, validar entradas, predecir y explicar.

No depende de Streamlit ni del paquete ``desercion``: solo del Pipeline serializado y de su
contrato (``model_metadata.json``), para que la imagen Docker sea mínima.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_bundle(artifact_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Carga el modelo y su contrato, y verifica que coincidan."""
    path = Path(artifact_dir)
    model = joblib.load(path / "model.joblib")
    metadata = json.loads((path / "model_metadata.json").read_text(encoding="utf-8"))
    if list(model.feature_names_in_) != metadata["feature_order"]:
        raise ValueError("El orden de variables del contrato no coincide con el del modelo")
    return model, metadata


def add_engineered(frame: pd.DataFrame) -> pd.DataFrame:
    """Variable construida, con la misma fórmula del entrenamiento (src/desercion/features.py)."""
    frame = frame.copy()
    frame["brecha.promedio"] = frame["average.first.period"] - frame["PNA"]
    return frame


def validate_frame(raw: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Valida una tabla de estudiantes contra el contrato (una fila por estudiante).

    Devuelve las filas con el orden de columnas del modelo y, por fila, el texto de los errores
    (vacío si la fila es válida). Reglas:

    - Todas las variables del contrato deben existir como columnas (si falta una, se rechaza
      el archivo completo). Columnas adicionales, como un identificador, se ignoran.
    - Numéricas: número finito dentro del rango visto en entrenamiento; celda vacía solo si la
      variable admite dato faltante.
    - Categóricas: una de las categorías vistas en entrenamiento.
    """
    contract = metadata["input_contract"]
    missing = [f for f in metadata["input_features"] if f not in raw.columns]
    if missing:
        raise ValueError(f"Faltan variables: {missing}")
    raw = raw.reset_index(drop=True)
    errors = pd.Series([[] for _ in range(len(raw))], dtype=object)
    clean = pd.DataFrame(index=raw.index)

    def flag(mask: pd.Series, message) -> None:
        for i in mask[mask].index:
            errors[i].append(message(i))

    for name in metadata["input_features"]:
        spec, col = contract[name], raw[name]
        empty = col.isna() | (col.astype(str).str.strip() == "")
        if spec["type"] == "categorical":
            text = col.astype(str).str.strip()
            flag(empty, lambda i, n=name: f"{n}: el dato es obligatorio")
            flag(~empty & ~text.isin(spec["categories"]),
                 lambda i, n=name, t=text: f"{n}: categoría desconocida {t[i]!r}")
            clean[name] = text.where(~empty)
            continue
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            col = col.astype(str).str.strip().str.replace(",", ".", regex=False)  # 88,5 -> 88.5
        num = pd.to_numeric(col.where(~empty), errors="coerce").astype(float)
        flag(~empty & num.isna(), lambda i, n=name, c=col: f"{n}: {c[i]!r} no es un número")
        if not spec["allow_missing"]:
            flag(empty, lambda i, n=name: f"{n}: el dato es obligatorio")
        out = num.notna() & ~num.between(spec["minimum"], spec["maximum"])
        flag(out, lambda i, n=name, s=spec, v=num: f"{n}: {v[i]} está fuera del rango "
                                                   f"[{s['minimum']}, {s['maximum']}]")
        clean[name] = num
    frame = add_engineered(clean)[metadata["feature_order"]]
    return frame, errors.map("; ".join)


def make_frame(values: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    """Una fila (un estudiante) con el orden del modelo; error si el contrato no la admite."""
    missing = [f for f in metadata["input_features"] if f not in values]
    if missing:
        raise ValueError(f"Faltan variables: {missing}")
    frame, errors = validate_frame(pd.DataFrame([values]), metadata)
    if errors[0]:
        raise ValueError(errors[0])
    return frame


def risk_percentile(probability, metadata: dict[str, Any]):
    """Porcentaje de estudiantes del desarrollo con un riesgo estimado menor que este."""
    pct = np.searchsorted(metadata["risk_percentiles"], probability, side="right")
    return int(pct) if np.ndim(pct) == 0 else pct


def example_batch(metadata: dict[str, Any], n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Estudiantes FICTICIOS para la plantilla de carga: se generan solo a partir del contrato.

    Cada numérica se sortea alrededor de su valor típico (distribución triangular entre el
    mínimo y el máximo) y cada categórica entre sus categorías, de forma independiente. No
    proviene de ningún registro real.
    """
    rng = np.random.default_rng(seed)
    rows = {"id_estudiante": [f"FICTICIO-{i + 1:03d}" for i in range(n)]}
    for name in metadata["input_features"]:
        spec = metadata["input_contract"][name]
        if spec["type"] == "categorical":
            # La mitad de las veces la categoría más frecuente; si no, una de las siguientes.
            others = [c for c in spec["categories"][:6] if c != spec["default"]] or [spec["default"]]
            rows[name] = np.where(rng.random(n) < 0.5, spec["default"], rng.choice(others, n))
            continue
        lo, mode, hi = spec["minimum"], spec["default"], spec["maximum"]
        # Rango acotado alrededor del valor típico para que los perfiles sean plausibles.
        lo, hi = max(lo, mode - (mode - lo) * 0.15), min(hi, mode + (hi - mode) * 0.15)
        values = rng.triangular(lo, mode, hi, n) if hi > lo else np.full(n, mode)
        rows[name] = np.round(values) if spec["integer"] else np.round(values, 2)
    return pd.DataFrame(rows)


# Niveles de riesgo, cada uno ligado a una acción (pocos niveles para evitar fatiga de alertas):
# - Alto: dentro del 20 % de mayor riesgo (en prueba, ese grupo concentra ~48 % de los desertores).
# - Medio: supera el umbral validado (F2) pero no está en el 20 % superior.
# - Bajo: por debajo del umbral.
TIERS = ("Alto", "Medio", "Bajo")


def risk_tier(probability, metadata: dict[str, Any]):
    """Nivel de riesgo (Alto, Medio o Bajo) de una probabilidad o de un arreglo de ellas."""
    p80 = metadata["risk_percentiles"][79]  # percentiles 1..99: el índice 79 es el 80
    thr = float(metadata["decision_threshold"])
    prob = np.asarray(probability, dtype=float)
    tier = np.where(prob >= p80, "Alto", np.where(prob >= thr, "Medio", "Bajo"))
    return str(tier) if tier.ndim == 0 else tier


def predict_batch(model: Any, metadata: dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    """Predice una tabla de estudiantes. Las filas inválidas no se predicen y reportan su error.

    Devuelve las columnas originales más probabilidad, nivel de riesgo, decisión, posición del
    riesgo y error.
    """
    frame, errors = validate_frame(raw, metadata)
    out = raw.reset_index(drop=True).copy()
    ok = errors == ""
    prob = pd.Series(np.nan, index=out.index)
    if ok.any():
        prob[ok] = model.predict_proba(frame[ok])[:, 1]
    threshold = float(metadata["decision_threshold"])
    out["probabilidad"] = prob
    out["nivel_riesgo"] = np.where(ok, risk_tier(prob.fillna(0).to_numpy(), metadata), "")
    out["alerta"] = np.where(ok, np.where(prob >= threshold, "Priorizar acompañamiento",
                                          "Sin alerta"), "")
    out["percentil_riesgo"] = pd.Series(risk_percentile(prob.fillna(0).to_numpy(), metadata),
                                        index=out.index).where(ok)
    out["error"] = errors
    return out


def predict(model: Any, metadata: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Probabilidad, nivel de riesgo, decisión con el umbral validado y posición relativa."""
    frame = make_frame(values, metadata)
    probability = float(model.predict_proba(frame)[0, 1])
    threshold = float(metadata["decision_threshold"])
    return {"probability": probability, "threshold": threshold,
            "prediction": int(probability >= threshold),
            "tier": risk_tier(probability, metadata),
            "percentile": risk_percentile(probability, metadata)}


def _labels(metadata: dict[str, Any]) -> dict[str, str]:
    labels = {f: spec["label"] for f, spec in metadata["input_contract"].items()}
    labels["brecha.promedio"] = "Brecha promedio 1.er sem. − bachillerato"
    return labels


def shap_by_variable(model: Any, metadata: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Aportes SHAP (log-odds) por variable original, una fila por estudiante ya validado.

    Se explica el clasificador sobre los datos transformados y se suman las columnas one-hot y
    la marca de faltante de cada variable para leerla como una sola.
    """
    import warnings

    import shap

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = prep.transform(frame)
    with warnings.catch_warnings():  # aviso informativo de SHAP sobre el formato de LightGBM
        warnings.simplefilter("ignore", UserWarning)
        raw = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
    if raw.ndim == 3:
        raw = raw[..., 1] if raw.shape[-1] == 2 else raw[1]
    contrib = pd.DataFrame(raw, columns=prep.get_feature_names_out(), index=frame.index)
    return contrib.T.groupby(pd.Series(metadata["shap_groups"])).sum().T


def local_shap_values(model: Any, metadata: dict[str, Any], values: dict[str, Any]) -> pd.DataFrame:
    """Aporte SHAP de cada variable a la predicción de un estudiante, de mayor a menor |aporte|."""
    by_var = shap_by_variable(model, metadata, make_frame(values, metadata)).iloc[0]
    out = pd.DataFrame({"variable": by_var.index, "etiqueta": by_var.index.map(_labels(metadata)),
                        "aporte_shap": by_var.to_numpy()})
    return out.reindex(out["aporte_shap"].abs().sort_values(ascending=False).index)


def group_drivers(model: Any, metadata: dict[str, Any], raw: pd.DataFrame,
                  top: int = 8) -> pd.DataFrame:
    """Variables que más pesan en las predicciones de un grupo (media del |SHAP| por variable)."""
    frame, errors = validate_frame(raw, metadata)
    frame = frame[errors == ""]
    if frame.empty:
        return pd.DataFrame(columns=["variable", "etiqueta", "importancia"])
    imp = shap_by_variable(model, metadata, frame).abs().mean().sort_values(ascending=False)
    imp = imp.head(top)
    return pd.DataFrame({"variable": imp.index, "etiqueta": imp.index.map(_labels(metadata)),
                         "importancia": imp.to_numpy()})
