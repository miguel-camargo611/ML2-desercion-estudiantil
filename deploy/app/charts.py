"""Gráficos Altair con el estilo del tablero (fondo transparente, marcas finas, tooltips)."""

from __future__ import annotations

import altair as alt
import pandas as pd
from theme import DOWN, INK, INK_2, INK_3, TIER_COLOR, UP, VIOLET

FONT = "Plus Jakarta Sans"
TIERS = ["Alto", "Medio", "Bajo"]
TIER_SCALE = alt.Scale(domain=TIERS, range=[TIER_COLOR[t] for t in TIERS])


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (chart.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont=FONT, titleFont=FONT, labelColor=INK_2, titleColor=INK_3,
                            labelFontSize=12, titleFontSize=11, labelLimit=240, gridColor="#ece8f6",
                            domainColor="#ddd7ee", tickColor="#ddd7ee")
            .configure_legend(labelFont=FONT, titleFont=FONT, labelColor=INK_2,
                              titleColor=INK_3, orient="bottom", symbolType="circle")
            .configure_text(font=FONT))


def risk_ring(percentile: int, probability: float, tier: str) -> alt.Chart:
    """Anillo: posición del estudiante entre todos (percentil); al centro, la probabilidad."""
    data = pd.DataFrame({"parte": ["riesgo", "resto"], "valor": [percentile, 100 - percentile]})
    ring = alt.Chart(data).mark_arc(innerRadius=78, outerRadius=100, cornerRadius=8,
                                    padAngle=0.02).encode(
        theta=alt.Theta("valor:Q", stack=True, sort=None),
        color=alt.Color("parte:N", scale=alt.Scale(domain=["riesgo", "resto"],
                                                   range=[TIER_COLOR[tier], "#ece8f6"]),
                        legend=None),
        order=alt.Order("parte:N", sort="descending"),
        tooltip=[alt.Tooltip("valor:Q", title="Percentil")])
    center = pd.DataFrame({"t": [f"{probability:.1%}".replace(".", ",")],
                           "s": ["probabilidad"]})
    big = alt.Chart(center).mark_text(fontSize=30, fontWeight=800, color=INK, dy=-6).encode(
        text="t:N")
    small = alt.Chart(center).mark_text(fontSize=12, color=INK_3, dy=20).encode(text="s:N")
    return _style(alt.layer(ring, big, small), 230)


def shap_bars(local: pd.DataFrame) -> alt.Chart:
    """Barras divergentes: qué variables suben (coral) o bajan (lavanda) el riesgo."""
    d = local.copy()
    d["efecto"] = d["aporte_shap"].map(lambda v: "Sube el riesgo" if v > 0 else "Baja el riesgo")
    d["orden"] = range(len(d))
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=16).encode(
        x=alt.X("aporte_shap:Q", title="Aporte al riesgo (SHAP, log-odds)",
                axis=alt.Axis(tickCount=5)),
        y=alt.Y("etiqueta:N", sort=alt.EncodingSortField("orden"), title=None),
        color=alt.Color("efecto:N", scale=alt.Scale(domain=["Sube el riesgo", "Baja el riesgo"],
                                                    range=[UP, DOWN]), title=None),
        tooltip=[alt.Tooltip("etiqueta:N", title="Variable"),
                 alt.Tooltip("aporte_shap:Q", title="Aporte", format="+.3f"),
                 alt.Tooltip("efecto:N", title="Efecto")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#cfc8e6").encode(x="x:Q")
    return _style(alt.layer(zero, bars), 34 * len(d) + 40)


def tier_donut(results: pd.DataFrame) -> alt.Chart:
    """Composición del grupo por nivel de riesgo (siempre en orden Alto, Medio, Bajo)."""
    counts = (results["nivel_riesgo"].value_counts().reindex(TIERS, fill_value=0)
              .rename_axis("nivel").reset_index(name="n"))
    counts["pct"] = counts["n"] / max(counts["n"].sum(), 1)
    counts["leyenda"] = [f"{t} · {n}" for t, n in zip(counts["nivel"], counts["n"], strict=True)]
    donut = alt.Chart(counts).mark_arc(innerRadius=62, outerRadius=96, cornerRadius=6,
                                       padAngle=0.03, stroke="white", strokeWidth=2).encode(
        theta=alt.Theta("n:Q", stack=True),
        color=alt.Color("nivel:N", scale=TIER_SCALE, title=None,
                        legend=alt.Legend(labelExpr="datum.label")),
        order=alt.Order("orden:Q"),
        tooltip=[alt.Tooltip("nivel:N", title="Nivel"), alt.Tooltip("n:Q", title="Estudiantes"),
                 alt.Tooltip("pct:Q", title="Porcentaje", format=".0%")],
    ).transform_calculate(orden="indexof(['Alto','Medio','Bajo'], datum.nivel)")
    total = pd.DataFrame({"t": [str(int(counts["n"].sum()))], "s": ["estudiantes"]})
    big = alt.Chart(total).mark_text(fontSize=26, fontWeight=800, color=INK, dy=-6).encode(
        text="t:N")
    small = alt.Chart(total).mark_text(fontSize=11, color=INK_3, dy=16).encode(text="s:N")
    return _style(alt.layer(donut, big, small), 260)


def probability_histogram(results: pd.DataFrame, prevalence: float) -> alt.Chart:
    """Distribución de la probabilidad estimada, coloreada por nivel; línea = tasa promedio."""
    d = results[results["error"] == ""][["probabilidad", "nivel_riesgo"]]
    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                 stroke="white", strokeWidth=1).encode(
        x=alt.X("probabilidad:Q", bin=alt.Bin(maxbins=24), title="Probabilidad estimada",
                axis=alt.Axis(format=".0%")),
        y=alt.Y("count():Q", title="Estudiantes", stack=True),
        color=alt.Color("nivel_riesgo:N", scale=TIER_SCALE, title=None),
        tooltip=[alt.Tooltip("nivel_riesgo:N", title="Nivel"),
                 alt.Tooltip("count():Q", title="Estudiantes")])
    rule = alt.Chart(pd.DataFrame({"x": [prevalence]})).mark_rule(
        color=INK_3, strokeDash=[4, 4]).encode(x="x:Q", tooltip=[
            alt.Tooltip("x:Q", title="Tasa promedio de deserción", format=".1%")])
    return _style(alt.layer(bars, rule), 260)


def importance_bars(df: pd.DataFrame, value: str, label: str = "etiqueta",
                    title: str = "Importancia (media |SHAP|)") -> alt.Chart:
    """Barras horizontales de un solo tono (magnitud), de mayor a menor."""
    d = df.copy()
    d["orden"] = range(len(d))
    chart = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=16, color=VIOLET).encode(
        x=alt.X(f"{value}:Q", title=title),
        y=alt.Y(f"{label}:N", sort=alt.EncodingSortField("orden"), title=None),
        tooltip=[alt.Tooltip(f"{label}:N", title="Variable"),
                 alt.Tooltip(f"{value}:Q", title="Importancia", format=".3f")])
    return _style(chart, 30 * len(d) + 40)
