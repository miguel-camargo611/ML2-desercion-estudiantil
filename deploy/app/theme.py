"""Tema visual: fondo pastel con degradado, tarjetas de vidrio esmerilado y piezas HTML.

Reglas de diseño (guías de glassmorphism para dashboards):
- El vidrio va en la navegación y en las tarjetas clave (encabezado, KPIs); los gráficos y
  tablas van sobre superficies casi opacas para que el texto pequeño se lea bien.
- El color de riesgo nunca va solo: cada nivel lleva su nombre escrito.
"""

from __future__ import annotations

import html

# Paleta (validada con el script de la guía de visualización: separación para daltonismo OK).
INK, INK_2, INK_3 = "#24213f", "#55527a", "#8a87a8"
VIOLET, VIOLET_2 = "#7c5cf0", "#a48bff"
TIER_COLOR = {"Alto": "#e8467c", "Medio": "#f08a3e", "Bajo": "#6d7cf0"}
TIER_SOFT = {"Alto": "#fde4ec", "Medio": "#feeedd", "Bajo": "#e6e9fd"}
# Acción corta (tarjeta) y completa (texto).
TIER_SHORT = {"Alto": "Contactar", "Medio": "Seguimiento", "Bajo": "Habitual"}
TIER_ACTION = {
    "Alto": "Contactar esta semana",
    "Medio": "Seguimiento en el mes",
    "Bajo": "Acompañamiento habitual",
}
UP, DOWN = TIER_COLOR["Alto"], TIER_COLOR["Bajo"]  # aporte SHAP que sube o baja el riesgo

CSS = f"""
<style>
/* Fondo: malla de degradados pastel (durazno, rosa, lavanda, cielo) */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1100px 620px at 8% -8%, #ffd8c4 0%, transparent 60%),
    radial-gradient(900px 600px at 95% 5%, #d9ccff 0%, transparent 58%),
    radial-gradient(900px 700px at 70% 105%, #f9c6de 0%, transparent 55%),
    radial-gradient(800px 600px at 0% 100%, #c8dafe 0%, transparent 55%),
    #f5f0fb;
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ color: {INK} !important; letter-spacing: -0.02em; }}
p, li, label, .stMarkdown {{ color: {INK_2}; }}

/* Barra lateral de vidrio */
[data-testid="stSidebar"] {{
  background: rgba(255,255,255,0.38) !important;
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-right: 1px solid rgba(255,255,255,0.7);
}}

/* Tarjetas: contenedores con key "glass-*" (vidrio) o "card-*" (casi opacas, para datos) */
[class*="st-key-glass"], [class*="st-key-card"] {{
  border-radius: 24px; padding: 1.25rem 1.35rem 1.1rem;
  border: 1px solid rgba(255,255,255,0.75);
  box-shadow: 0 10px 30px rgba(92, 72, 170, 0.10), inset 0 1px 0 rgba(255,255,255,0.8);
}}
[class*="st-key-glass"] {{
  background: rgba(255,255,255,0.42);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}}
[class*="st-key-card"] {{ background: rgba(255,255,255,0.82); }}

/* Pestañas como control segmentado */
[data-testid="stTabs"] [role="tablist"] {{
  gap: 6px; padding: 6px; border-radius: 999px; width: fit-content;
  background: rgba(255,255,255,0.45); border: 1px solid rgba(255,255,255,0.75);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}}
[data-testid="stTabs"] [role="tab"] {{
  border-radius: 999px; padding: 0.45rem 1.1rem; color: {INK_2}; font-weight: 600;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, {VIOLET} 0%, {VIOLET_2} 100%); color: white;
  box-shadow: 0 6px 16px rgba(124, 92, 240, 0.35);
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color: white; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

/* Botones */
button[data-testid^="stBaseButton-primary"], button[data-testid^="stBaseButton-primary"] p {{
  color: white !important;
}}
button[data-testid^="stBaseButton-primary"] {{
  background: linear-gradient(135deg, {VIOLET} 0%, #b07cf7 55%, #f08ab0 100%) !important;
  border: none !important; border-radius: 14px !important; font-weight: 700 !important;
  box-shadow: 0 8px 20px rgba(124, 92, 240, 0.30);
}}
button[data-testid^="stBaseButton-secondary"] {{
  border-radius: 14px !important; background: rgba(255,255,255,0.7) !important;
  border: 1px solid rgba(124, 92, 240, 0.25) !important; color: {INK} !important;
}}

/* Formularios, expanders y carga de archivo */
[data-testid="stForm"] {{ border: none; padding: 0; }}
[data-testid="stExpander"] details {{
  background: rgba(255,255,255,0.7); border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.85);
}}
[data-testid="stFileUploaderDropzone"] {{
  background: rgba(255,255,255,0.6); border: 2px dashed rgba(124, 92, 240, 0.35);
  border-radius: 18px;
}}
[data-testid="stDataFrameResizable"] {{ border-radius: 16px; overflow: hidden; }}

/* Piezas HTML propias */
.hero-title {{ font-size: 2.05rem; font-weight: 800; color: {INK}; margin: 0;
  letter-spacing: -0.03em; line-height: 1.15; }}
.hero-sub {{ color: {INK_2}; margin: .35rem 0 .9rem; font-size: 1rem; }}
.chip {{ display: inline-flex; align-items: center; gap: .35rem; padding: .32rem .75rem;
  margin: 0 .4rem .4rem 0; border-radius: 999px; font-size: .8rem; font-weight: 600;
  background: rgba(255,255,255,0.7); color: {INK_2}; border: 1px solid rgba(255,255,255,0.9); }}
.chip b {{ color: {INK}; }}
.kpi-label {{ font-size: .78rem; font-weight: 600; color: {INK_3}; text-transform: uppercase;
  letter-spacing: .06em; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 800; color: {INK}; line-height: 1.15;
  margin-top: .25rem; letter-spacing: -0.02em; }}
.kpi-sub {{ font-size: .82rem; color: {INK_2}; margin-top: .2rem; }}
.kpi-icon {{ width: 38px; height: 38px; border-radius: 12px; display: flex;
  align-items: center; justify-content: center; font-size: 1.15rem; margin-bottom: .55rem;
  color: white; box-shadow: 0 6px 14px rgba(92, 72, 170, 0.20); }}
.badge {{ display: inline-block; padding: .22rem .7rem; border-radius: 999px;
  font-weight: 700; font-size: .82rem; }}
.section-title {{ font-size: 1.05rem; font-weight: 700; color: {INK}; margin: 0 0 .15rem; }}
.section-sub {{ font-size: .84rem; color: {INK_3}; margin: 0 0 .6rem; }}
.factor {{ display: flex; justify-content: space-between; align-items: center;
  padding: .45rem .7rem; border-radius: 12px; margin-bottom: .35rem; font-size: .88rem;
  background: rgba(255,255,255,0.75); color: {INK}; }}
.empty {{ text-align: center; padding: 2.2rem 1rem; color: {INK_3}; }}
.empty .big {{ font-size: 2.4rem; }}
.note {{ font-size: .8rem; color: {INK_3}; line-height: 1.45; margin-top: .4rem; }}
</style>
"""

_ICON_BG = {
    "violet": f"linear-gradient(135deg, {VIOLET}, {VIOLET_2})",
    "pink": "linear-gradient(135deg, #e8467c, #f590b5)",
    "peach": "linear-gradient(135deg, #f08a3e, #f8b878)",
    "blue": "linear-gradient(135deg, #5b7cf0, #8fb0ff)",
}


def esc(text) -> str:
    return html.escape(str(text))


def hero(title: str, subtitle: str, chips: list[tuple[str, str]]) -> str:
    chip_html = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in chips)
    return (f'<div class="hero-title">{esc(title)}</div><div class="hero-sub">{esc(subtitle)}</div>'
            f"<div>{chip_html}</div>")


def kpi(label: str, value: str, sub: str = "", icon: str = "", tone: str = "violet") -> str:
    icon_html = (f'<div class="kpi-icon" style="background:{_ICON_BG[tone]}">{icon}</div>'
                 if icon else "")
    return (f'{icon_html}<div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{esc(value)}</div><div class="kpi-sub">{esc(sub)}</div>')


def badge(tier: str) -> str:
    return (f'<span class="badge" style="background:{TIER_SOFT[tier]};color:{INK};'
            f'border:1.5px solid {TIER_COLOR[tier]}">● Riesgo {esc(tier.lower())}</span>')


def section(title: str, sub: str = "") -> str:
    return (f'<div class="section-title">{esc(title)}</div>'
            + (f'<div class="section-sub">{esc(sub)}</div>' if sub else ""))


def factor_rows(local, n_up: int = 3, n_down: int = 2) -> str:
    """Factores en palabras: los que más suben y más bajan el riesgo de este estudiante."""
    ups = local[local["aporte_shap"] > 0].head(n_up)
    downs = local[local["aporte_shap"] < 0].head(n_down)
    rows = [f'<div class="factor"><span>▲ {esc(r.etiqueta)}</span>'
            f'<span style="color:{UP};font-weight:700">sube</span></div>' for r in ups.itertuples()]
    rows += [f'<div class="factor"><span>▼ {esc(r.etiqueta)}</span>'
             f'<span style="color:{DOWN};font-weight:700">baja</span></div>'
             for r in downs.itertuples()]
    return "".join(rows)


def empty_state(icon: str, text: str) -> str:
    return f'<div class="empty"><div class="big">{icon}</div><div>{esc(text)}</div></div>'
