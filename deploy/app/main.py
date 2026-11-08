"""App Streamlit: riesgo de deserción al terminar el primer semestre (demostración académica)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from model_service import (
    example_batch,
    load_bundle,
    local_shap_values,
    predict,
    predict_batch,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"

# Secciones del formulario (variables del contrato agrupadas como las vería un consejero).
SECTIONS = {
    "Desempeño en el primer semestre": ["average.first.period", "failed.subject.first.period",
                                        "dropped.subject.first.period", "FTE"],
    "Admisión": ["PNA", "admission.test", "admission.rubric", "general.math.eval",
                 "english.evaluation"],
    "Beca y apoyo financiero": ["scholarship.perc", "loan.perc", "scholarship.type"],
    "Programa y campus": ["school", "program", "region"],
    "Perfil y familia": ["age", "gender", "school.cost", "max.degree.parents",
                         "father.education.summary", "mother.education.summary", "parents.exatec"],
}
PERCENT = {"scholarship.perc", "loan.perc"}  # se guardan como fracción 0–1 y se muestran en %
RED, BLUE = "#e34948", "#2a78d6"

# Traducción solo para mostrar; al modelo llega el valor original del dataset.
ES = {
    "Male": "Masculino", "Female": "Femenino", "Yes": "Sí", "No": "No",
    "No information": "Sin información", "No degree": "Sin título universitario",
    "Undergraduate degree": "Pregrado", "Master degree": "Maestría", "PhD": "Doctorado",
    "High cost": "Costo alto", "Medium high cost": "Costo medio-alto", "Medium cost": "Costo medio",
    "Low cost": "Costo bajo", "Public": "Público", "Not defined": "No definido",
    "No scholarship": "Sin beca", "Traditional": "Tradicional",
    "Academic talent": "Talento académico", "Sports talent": "Talento deportivo",
    "Cultural talent": "Talento cultural", "Leadership talent": "Talento de liderazgo",
    "Enterpreneurial talent": "Talento emprendedor",
    "Leaders of Tomorrow Scholarship": "Líderes del Mañana",
    "Child of Professor/Employee/Director": "Hijo(a) de profesor o empleado",
    "Army/Navy scholarship": "Beca Ejército/Marina",
    "EIC": "Ingeniería y Ciencias (EIC)", "EN": "Negocios (EN)",
    "ECSG": "Ciencias Sociales y Gobierno (ECSG)", "EMCS": "Medicina y Ciencias de la Salud (EMCS)",
    "EHE-EAAD": "Humanidades y Educación / Arquitectura, Arte y Diseño (EHE-EAAD)",
    "EAAD-Engineering and Sciences": "Arquitectura, Arte y Diseño – Ingeniería (EAAD)",
    "RM": "Monterrey (RM)", "RCM": "Ciudad de México (RCM)", "RCS": "Centro-Sur (RCS)",
    "RO": "Occidente (RO)",
}


@st.cache_resource
def get_bundle():
    """Carga los artefactos una sola vez por proceso."""
    return load_bundle(ARTIFACT_DIR)


# ---------------------------------------------------------------- piezas de interfaz
def numeric_input(name: str, spec: dict) -> float | None:
    if name in PERCENT:
        value = st.number_input(f"{spec['label']} (%)", 0.0, round(spec["maximum"] * 100, 1),
                                round(spec["default"] * 100, 1), step=5.0, key=name)
        return value / 100
    if spec["allow_missing"] and st.checkbox(f"{spec['label']}: sin dato", key=f"{name}-na"):
        return None
    lo, hi, default = spec["minimum"], spec["maximum"], spec["default"]
    if spec["integer"]:
        return float(st.number_input(spec["label"], int(lo), int(hi), int(default), step=1,
                                     key=name))
    return float(st.number_input(spec["label"], float(lo), float(hi), float(default),
                                 step=round(max((hi - lo) / 100, 0.01), 2), key=name))


def categorical_input(name: str, spec: dict) -> str:
    options = spec["categories"]
    return st.selectbox(spec["label"], options, index=options.index(spec["default"]),
                        format_func=lambda v: ES.get(v, v), key=name)


def show_explanation(values: dict) -> None:
    """Gráfico SHAP local: qué variables empujan el riesgo de este estudiante."""
    try:
        local = local_shap_values(model, metadata, values).head(8)
    except Exception as error:  # en una app docente es preferible ver el diagnóstico
        st.warning(f"No fue posible calcular la explicación: {error}")
        return
    chart = pd.DataFrame({
        "Aumenta el riesgo": local["aporte_shap"].clip(lower=0).to_numpy(),
        "Reduce el riesgo": local["aporte_shap"].clip(upper=0).to_numpy(),
    }, index=local["etiqueta"])
    st.bar_chart(chart, horizontal=True, color=[RED, BLUE], sort=False)
    st.caption("Aporte de cada variable (SHAP, en log-odds) respecto al estudiante promedio. "
               "Describe cómo decide el modelo; no es una relación causal.")


def read_upload(upload) -> pd.DataFrame:
    """CSV (separador , o ; detectado solo) o Excel; todo se lee como texto y se valida después."""
    if upload.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(upload, dtype=str)
    return pd.read_csv(upload, sep=None, engine="python", dtype=str, encoding="utf-8-sig")


def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig: Excel muestra bien las tildes


def column_guide() -> pd.DataFrame:
    rows = []
    for name in metadata["input_features"]:
        spec = contract[name]
        if spec["type"] == "categorical":
            allowed = ", ".join(spec["categories"])
        else:
            allowed = f"{spec['minimum']:g} a {spec['maximum']:g}"
            if name in PERCENT:
                allowed += " (fracción: 0.4 = 40 %)"
            if spec["allow_missing"]:
                allowed += "; puede ir vacío"
        rows.append({"columna": name, "significado": spec["label"], "valores admitidos": allowed})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- página
st.set_page_config(page_title="Riesgo de deserción · ML II", page_icon="🎓", layout="wide")
model, metadata = get_bundle()
contract = metadata["input_contract"]
m = metadata["metrics"]

st.title("Riesgo de deserción al terminar el primer semestre")
st.caption(f"{metadata['model_name']} · versión {metadata['model_version']} · "
           "demostración académica, no para decisiones reales")

with st.sidebar:
    st.header("Trazabilidad")
    st.write(f"**Datos:** {metadata['dataset']}")
    st.write(f"**Entrenado:** {metadata['trained_at_utc'][:10]}")
    st.write(f"**Umbral validado:** {metadata['decision_threshold']:.3f} (elegido con F2)")
    st.subheader("Desempeño en prueba")
    st.metric("PR-AUC", f"{m['pr_auc']:.3f}", help=f"Azar = {m['prevalencia']:.3f}")
    st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.metric("Recall con el umbral", f"{m['recall']:.0%}")
    st.metric("Desertores en el 20 % de mayor riesgo", f"{m['captura_top20']:.0%}")

tab_one, tab_batch, tab_model = st.tabs(["Un estudiante", "Carga de archivo (uno o varios)",
                                         "Sobre el modelo"])

# ------------------------------------------------------------- 1. un estudiante
with tab_one:
    values: dict[str, object] = {}
    with st.form("perfil"):
        st.subheader("Datos del estudiante")
        for title, fields in SECTIONS.items():
            with st.expander(title, expanded=title.startswith("Desempeño")):
                cols = st.columns(3)
                for i, name in enumerate(fields):
                    with cols[i % 3]:
                        spec = contract[name]
                        values[name] = (categorical_input(name, spec)
                                        if spec["type"] == "categorical"
                                        else numeric_input(name, spec))
        submitted = st.form_submit_button("Estimar riesgo", type="primary",
                                          use_container_width=True)
    if submitted:
        try:
            result = predict(model, metadata, values)
        except ValueError as error:
            st.error(f"Entrada no válida: {error}")
            st.stop()
        st.subheader("Resultado")
        a, b, c = st.columns(3)
        a.metric("Probabilidad estimada de deserción", f"{result['probability']:.1%}",
                 help=f"Tasa promedio en los datos: {m['prevalencia']:.1%}")
        b.metric("Decisión con el umbral validado",
                 "Priorizar acompañamiento" if result["prediction"] else "Sin alerta")
        c.metric("Posición del riesgo", f"Mayor que el {result['percentile']} % de estudiantes")
        st.subheader("¿Por qué? Explicación de esta predicción")
        show_explanation(values)

# ------------------------------------------------------------- 2. carga de archivo
with tab_batch:
    st.subheader("Evaluar estudiantes desde un archivo")
    st.write("Suba un **CSV** o **Excel** con una fila por estudiante y una columna por "
             "variable (los nombres de columna deben ser exactamente los de la plantilla). "
             "Puede incluir una columna `id_estudiante` para identificarlos. Sirve para uno o "
             "para muchos estudiantes. El archivo se procesa en memoria y no se guarda.")
    d1, d2 = st.columns(2)
    template = example_batch(metadata, n=20)
    d1.download_button("Descargar plantilla vacía (CSV)", to_csv(template.head(0)),
                       "plantilla_estudiantes.csv", "text/csv", use_container_width=True)
    d2.download_button("Descargar ejemplo con 20 estudiantes ficticios (CSV)", to_csv(template),
                       "ejemplo_estudiantes_ficticios.csv", "text/csv", use_container_width=True)
    with st.expander("Columnas y valores admitidos"):
        st.dataframe(column_guide(), hide_index=True, use_container_width=True)

    upload = st.file_uploader("Archivo de estudiantes", type=["csv", "xlsx"])
    if upload is not None:
        try:
            raw = read_upload(upload)
            results = predict_batch(model, metadata, raw)
        except ValueError as error:
            st.error(f"El archivo no se puede evaluar: {error}")
            st.stop()

        ok = results["error"] == ""
        alerts = results["alerta"] == "Priorizar acompañamiento"
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estudiantes evaluados", f"{ok.sum():,}".replace(",", "."))
        k2.metric("Filas con error", int((~ok).sum()))
        k3.metric("Con alerta", f"{alerts.sum()} ({alerts.sum() / max(ok.sum(), 1):.0%})")
        k4.metric("Probabilidad media", f"{results.loc[ok, 'probabilidad'].mean():.1%}"
                  if ok.any() else "—")

        id_col = "id_estudiante" if "id_estudiante" in results else None
        if id_col is None:
            results.insert(0, "fila", range(2, len(results) + 2))  # fila como se ve en Excel
            id_col = "fila"
        ranked = results[ok].sort_values("probabilidad", ascending=False)
        shown = [id_col, "probabilidad", "alerta", "percentil_riesgo", "average.first.period",
                 "scholarship.perc", "program"]
        st.dataframe(
            ranked[shown], hide_index=True, use_container_width=True,
            column_config={
                "probabilidad": st.column_config.ProgressColumn(
                    "Probabilidad", format="percent", min_value=0.0, max_value=1.0),
                "percentil_riesgo": st.column_config.NumberColumn(
                    "Riesgo mayor que el … % de estudiantes", format="%d %%"),
                "alerta": "Decisión",
                "average.first.period": "Promedio 1.er sem.",
                "scholarship.perc": st.column_config.NumberColumn("% de beca", format="percent"),
                "program": "Programa",
            })
        st.download_button("Descargar resultados (CSV)", to_csv(results),
                           "resultados_riesgo.csv", "text/csv")

        if (~ok).any():
            with st.expander(f"Filas con error ({int((~ok).sum())})"):
                st.dataframe(results.loc[~ok, [id_col, "error"]], hide_index=True,
                             use_container_width=True)

        if ok.any():
            st.subheader("Explicación de un estudiante de la lista")
            chosen = st.selectbox("Estudiante", ranked[id_col].astype(str).tolist())
            row = ranked[ranked[id_col].astype(str) == chosen].iloc[0]
            st.write(f"Probabilidad **{row['probabilidad']:.1%}** · {row['alerta']} · riesgo "
                     f"mayor que el {int(row['percentil_riesgo'])} % de estudiantes")
            show_explanation(raw.reset_index(drop=True).loc[row.name,
                                                            metadata["input_features"]].to_dict())

# ------------------------------------------------------------- 3. sobre el modelo
with tab_model:
    st.subheader("Variables más influyentes en general")
    imp = pd.DataFrame(metadata["global_importance"]).head(10)
    st.bar_chart(imp.set_index("label")["importance"], horizontal=True, color=BLUE, sort=False)
    st.caption("Promedio del |SHAP| por variable en el conjunto de desarrollo.")
    st.subheader("Limitaciones y uso responsable")
    for limitation in metadata["limitations"]:
        st.write(f"- {limitation}")
