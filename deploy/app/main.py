"""App Streamlit: riesgo de deserción al terminar el primer semestre (demostración académica)."""

from __future__ import annotations

from pathlib import Path

import charts
import pandas as pd
import streamlit as st
import theme
from model_service import (
    example_batch,
    group_drivers,
    load_bundle,
    local_shap_values,
    predict,
    predict_batch,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"

# Secciones del formulario (variables del contrato agrupadas como las vería un consejero).
SECTIONS = {
    "📚 Desempeño en el primer semestre": ["average.first.period", "failed.subject.first.period",
                                           "dropped.subject.first.period", "FTE"],
    "📝 Admisión": ["PNA", "admission.test", "admission.rubric", "general.math.eval",
                   "english.evaluation"],
    "💳 Beca y apoyo financiero": ["scholarship.perc", "loan.perc", "scholarship.type"],
    "🏛️ Programa y campus": ["school", "program", "region"],
    "👤 Perfil y familia": ["age", "gender", "school.cost", "max.degree.parents",
                           "father.education.summary", "mother.education.summary",
                           "parents.exatec"],
}
PERCENT = {"scholarship.perc", "loan.perc"}  # se guardan como fracción 0–1 y se muestran en %
TIER_EMOJI = {"Alto": "🔴 Alto", "Medio": "🟠 Medio", "Bajo": "🔵 Bajo"}

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


def pct(x: float, d: int = 1) -> str:
    """Porcentaje con coma decimal."""
    return f"{x:.{d}%}".replace(".", ",")


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def altair(chart) -> None:
    st.altair_chart(chart, width="stretch", theme=None)


@st.cache_resource
def get_bundle():
    """Carga los artefactos una sola vez por proceso."""
    return load_bundle(ARTIFACT_DIR)


# ---------------------------------------------------------------- entradas
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


# ---------------------------------------------------------------- bloques de resultado
def student_result(values: dict, result: dict, key: str) -> None:
    """Tarjeta de resultado de un estudiante: nivel, anillo de riesgo, acción y factores."""
    tier = result["tier"]
    with st.container(key=f"glass-result-{key}"):
        html(theme.section("Resultado") + theme.badge(tier))
        altair(charts.risk_ring(result["percentile"], result["probability"], tier))
        a, b = st.columns(2)
        with a:
            html(theme.kpi("Percentil de riesgo", str(result["percentile"]),
                           f"mayor que el {result['percentile']} % de estudiantes"))
        with b:
            html(theme.kpi("Acción sugerida", theme.TIER_SHORT[tier],
                           theme.TIER_ACTION[tier]))
        html(f'<div class="note">Promedio de deserción en los datos: {pct(m["prevalencia"])} · '
             f'umbral validado: {pct(metadata["decision_threshold"])}</div>')
    with st.container(key=f"card-why-{key}"):
        html(theme.section("¿Por qué este resultado?",
                           "Variables que más empujan el riesgo de este estudiante"))
        try:
            local = local_shap_values(model, metadata, values).head(8)
        except Exception as error:  # en una app docente es preferible ver el diagnóstico
            st.warning(f"No fue posible calcular la explicación: {error}")
            return
        html(theme.factor_rows(local))
        altair(charts.shap_bars(local))
        html('<div class="note">SHAP describe cómo decide el modelo respecto al estudiante '
             "promedio; no es una relación causal.</div>")


# ---------------------------------------------------------------- página
st.set_page_config(page_title="Riesgo de deserción · ML II", page_icon="🎓", layout="wide")
html(theme.CSS)
model, metadata = get_bundle()
contract = metadata["input_contract"]
m = metadata["metrics"]

with st.container(key="glass-hero"):
    html(theme.hero(
        "Alerta temprana de deserción",
        "Riesgo de no continuar al segundo año, estimado al terminar el primer semestre.",
        [("Modelo", f"LightGBM v{metadata['model_version']}"),
         ("PR-AUC en prueba", f"{m['pr_auc']:.3f}".replace(".", ",")),
         ("Detecta en el top 20 %", f"{pct(m['captura_top20'], 0)} de desertores"),
         ("Datos", "Tec de Monterrey · AD19–AD20")]))

with st.sidebar:
    html(theme.section("Niveles de riesgo", "Cada nivel tiene una acción"))
    for tier, rule in [("Alto", "20 % de estudiantes con mayor riesgo"),
                       ("Medio", "supera el umbral validado (F2)"),
                       ("Bajo", "por debajo del umbral")]:
        html(f'<div style="margin:.55rem 0">{theme.badge(tier)}'
             f'<div class="note" style="margin-top:.25rem">{rule} · '
             f"<b>{theme.TIER_ACTION[tier]}</b></div></div>")
    st.divider()
    html(theme.section("Trazabilidad"))
    html(f'<div class="note">Entrenado el {metadata["trained_at_utc"][:10]}<br>'
         f'Umbral F2: {metadata["decision_threshold"]:.3f}<br>'
         f"Demostración académica: no usar para decisiones reales.</div>")

tab_one, tab_batch, tab_model = st.tabs(["👤  Un estudiante", "📂  Carga de archivo",
                                         "📊  Sobre el modelo"])

# ------------------------------------------------------------- 1. un estudiante
with tab_one:
    left, right = st.columns([1.1, 1], gap="large")
    with left, st.container(key="card-form"):
        html(theme.section("Datos del estudiante",
                           "Valores típicos precargados; ajuste los que conozca."))
        values: dict[str, object] = {}
        with st.form("perfil", border=False):
            for title, fields in SECTIONS.items():
                with st.expander(title, expanded=title.startswith("📚")):
                    cols = st.columns(2)
                    for i, name in enumerate(fields):
                        with cols[i % 2]:
                            spec = contract[name]
                            values[name] = (categorical_input(name, spec)
                                            if spec["type"] == "categorical"
                                            else numeric_input(name, spec))
            submitted = st.form_submit_button("Estimar riesgo", type="primary",
                                              width="stretch")
    if submitted:
        try:
            st.session_state["single"] = (values, predict(model, metadata, values))
        except ValueError as error:
            st.session_state.pop("single", None)
            left.error(f"Entrada no válida: {error}")
    with right:
        if "single" in st.session_state:
            student_result(*st.session_state["single"], key="one")
        else:
            with st.container(key="glass-empty"):
                html(theme.empty_state("✨", "Complete los datos y pulse «Estimar riesgo» para "
                                             "ver el nivel de riesgo y su explicación."))

# ------------------------------------------------------------- 2. carga de archivo
with tab_batch:
    with st.container(key="card-upload"):
        html(theme.section("Evaluar estudiantes desde un archivo",
                           "CSV o Excel, una fila por estudiante (uno o varios). Se procesa "
                           "en memoria y no se guarda."))
        up, d1, d2 = st.columns([2, 1, 1], vertical_alignment="bottom")
        with up:
            upload = st.file_uploader("Archivo de estudiantes", type=["csv", "xlsx"],
                                      label_visibility="collapsed")
        template = example_batch(metadata, n=20)
        d1.download_button("⬇ Plantilla vacía", to_csv(template.head(0)),
                           "plantilla_estudiantes.csv", "text/csv", width="stretch")
        d2.download_button("⬇ Ejemplo ficticio (20)", to_csv(template),
                           "ejemplo_estudiantes_ficticios.csv", "text/csv",
                           width="stretch")
        with st.expander("Columnas y valores admitidos"):
            st.dataframe(column_guide(), hide_index=True, width="stretch")

    if upload is None:
        with st.container(key="glass-empty-batch"):
            html(theme.empty_state("📂", "Suba un archivo para ver el tablero del grupo. "
                                         "¿Sin datos a mano? Descargue el ejemplo ficticio."))
    else:
        try:
            raw = read_upload(upload)
            results = predict_batch(model, metadata, raw)
        except ValueError as error:
            st.error(f"El archivo no se puede evaluar: {error}")
            st.stop()
        ok = results["error"] == ""
        n_ok = int(ok.sum())
        high = int((results["nivel_riesgo"] == "Alto").sum())
        alerts = int((results["alerta"] == "Priorizar acompañamiento").sum())
        mean_p = results.loc[ok, "probabilidad"].mean() if n_ok else float("nan")

        k = st.columns(4)
        cards = [
            ("Evaluados", f"{n_ok:,}".replace(",", "."), f"de {len(results)} filas", "👥",
             "violet"),
            ("Riesgo alto", str(high), f"{pct(high / max(n_ok, 1), 0)} del grupo · contactar ya",
             "🔥", "pink"),
            ("Con alerta", str(alerts), f"alto + medio · {pct(alerts / max(n_ok, 1), 0)} del grupo",
             "🔔", "peach"),
            ("Probabilidad media", pct(mean_p) if n_ok else "—",
             f"promedio histórico {pct(m['prevalencia'])}", "📈", "blue"),
        ]
        for col, (label, value, sub, icon, tone) in zip(k, cards, strict=True):
            with col, st.container(key=f"glass-kpi-{tone}"):
                html(theme.kpi(label, value, sub, icon, tone))

        if n_ok:
            c1, c2 = st.columns([1, 1.6], gap="medium")
            with c1, st.container(key="card-donut"):
                html(theme.section("Composición por nivel", "Estudiantes en cada nivel"))
                altair(charts.tier_donut(results[ok]))
            with c2, st.container(key="card-hist"):
                html(theme.section("Distribución del riesgo",
                                   "Probabilidad estimada; la línea punteada es la tasa promedio"))
                altair(charts.probability_histogram(results, m["prevalencia"]))

            id_col = "id_estudiante" if "id_estudiante" in results else None
            if id_col is None:
                results.insert(0, "fila", range(2, len(results) + 2))  # fila como en Excel
                id_col = "fila"
            ranked = results[ok].sort_values("probabilidad", ascending=False).copy()
            ranked["nivel"] = ranked["nivel_riesgo"].map(TIER_EMOJI)
            ranked["accion"] = ranked["nivel_riesgo"].map(theme.TIER_SHORT)

            w1, w2 = st.columns([1.6, 1], gap="medium")
            with w1, st.container(key="card-watchlist"):
                html(theme.section("Lista priorizada", "Ordenada de mayor a menor riesgo"))
                st.dataframe(
                    ranked[[id_col, "nivel", "probabilidad", "percentil_riesgo", "accion",
                            "average.first.period", "program"]],
                    hide_index=True, width="stretch", height=360,
                    column_config={
                        id_col: "Estudiante",
                        "nivel": "Nivel",
                        # Escala al máximo del grupo: con probabilidades de ~5–15 % una barra
                        # de 0 a 100 % casi no se vería.
                        "probabilidad": st.column_config.ProgressColumn(
                            "Probabilidad", format="percent", min_value=0.0,
                            max_value=float(max(ranked["probabilidad"].max(), 0.01))),
                        "percentil_riesgo": st.column_config.NumberColumn(
                            "Percentil", format="%d"),
                        "accion": "Acción sugerida",
                        "average.first.period": "Promedio 1.er sem.",
                        "program": "Programa",
                    })
                st.download_button("⬇ Descargar resultados (CSV)", to_csv(results),
                                   "resultados_riesgo.csv", "text/csv")
            with w2, st.container(key="card-drivers"):
                html(theme.section("Qué pesa más en este grupo",
                                   "Media del |SHAP| por variable"))
                altair(charts.importance_bars(group_drivers(model, metadata, raw), "importancia"))

            with st.container(key="card-drill"):
                html(theme.section("Ficha de un estudiante", "Elija un estudiante de la lista"))
                chosen = st.selectbox("Estudiante", ranked[id_col].astype(str).tolist(),
                                      label_visibility="collapsed")
                row = ranked[ranked[id_col].astype(str) == chosen].iloc[0]
                vals = raw.reset_index(drop=True).loc[row.name,
                                                      metadata["input_features"]].to_dict()
                s1, s2 = st.columns([1, 1.3], gap="medium")
                with s1:
                    student_result(vals, predict(model, metadata, vals), key="batch")
                with s2:
                    html(theme.section("Datos registrados"))
                    shown = {contract[f]["label"]: ES.get(str(v), v) for f, v in vals.items()}
                    st.dataframe(pd.Series(shown, name="valor").astype(str), height=520,
                                 width="stretch")

        if (~ok).any():
            with st.expander(f"⚠️ Filas con error ({int((~ok).sum())})"):
                id_err = "id_estudiante" if "id_estudiante" in results else "fila"
                st.dataframe(results.loc[~ok, [id_err, "error"]], hide_index=True,
                             width="stretch")

# ------------------------------------------------------------- 3. sobre el modelo
with tab_model:
    k = st.columns(4)
    model_cards = [
        ("PR-AUC en prueba", f"{m['pr_auc']:.3f}".replace(".", ","),
         f"{m['pr_auc'] / m['prevalencia']:.1f} veces el azar ({m['prevalencia']:.3f})"
         .replace(".", ","), "🎯", "violet"),
        ("ROC-AUC", f"{m['roc_auc']:.3f}".replace(".", ","), "qué tan bien ordena",
         "📐", "blue"),
        ("Recall con el umbral", pct(m["recall"], 0), "desertores que reciben alerta", "🔔",
         "peach"),
        ("Top 20 % de riesgo", pct(m["captura_top20"], 0), "de los desertores quedan ahí", "🔥",
         "pink"),
    ]
    for col, (label, value, sub, icon, tone) in zip(k, model_cards, strict=True):
        with col, st.container(key=f"glass-model-{tone}"):
            html(theme.kpi(label, value, sub, icon, tone))

    g1, g2 = st.columns([1.3, 1], gap="medium")
    with g1, st.container(key="card-global"):
        html(theme.section("Variables más influyentes",
                           "Media del |SHAP| en el conjunto de desarrollo"))
        imp = pd.DataFrame(metadata["global_importance"]).head(10)
        altair(charts.importance_bars(imp, "importance", label="label"))
    with g2, st.container(key="card-limits"):
        html(theme.section("Limitaciones y uso responsable"))
        for limitation in metadata["limitations"]:
            html(f'<div class="factor" style="display:block">{theme.esc(limitation)}</div>')
        html('<div class="note">Métricas de la evaluación única en el conjunto de prueba '
             "sellado (20 % de los datos, nunca usado para entrenar ni para decidir).</div>")
