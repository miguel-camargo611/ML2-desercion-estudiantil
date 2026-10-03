"""Figuras del informe. Paleta categórica validada (azul, naranja, aqua) y gris para referencias."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from desercion.features import VAR_LABELS  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#9a9994", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = [BLUE, ORANGE, AQUA]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _num(v: float, d: int = 2) -> str:
    """Formato numérico en español (coma decimal)."""
    return f"{v:.{d}f}".replace(".", ",")


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- datos
def data_quality(q: pd.DataFrame, used: list[str], path: Path) -> Path:
    q = q[q.variable.isin(used)].copy()
    q["ausente"] = q["pct_nan"] + q["pct_no_information"]
    q = q[q["ausente"] > 1].sort_values("ausente")
    fig, ax = plt.subplots(figsize=(6.0, 0.22 * len(q) + 0.8))
    labels = [VAR_LABELS.get(v, v) for v in q.variable]
    ax.barh(labels, q["pct_nan"], color=BLUE, height=0.65, label="Vacío / no aplica (NaN)")
    ax.barh(labels, q["pct_no_information"], left=q["pct_nan"], color=ORANGE, height=0.65,
            label="Codificado 'No information'", edgecolor="white", linewidth=1)
    ax.set_xlabel("% de registros (AD19–AD20, desarrollo)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def selection_importance(imp: pd.DataFrame, dropped: list[str], path: Path) -> Path:
    """Importancia por permutación (media y rango en los cinco pliegues) y variables descartadas."""
    g = imp.groupby("variable")["importancia"].agg(["mean", "min", "max"]).sort_values("mean")
    y = np.arange(len(g))
    colors = [GRAY if v in dropped else BLUE for v in g.index]
    fig, ax = plt.subplots(figsize=(6.4, 0.2 * len(g) + 0.9))
    ax.hlines(y, g["min"], g["max"], color=colors, lw=1.5, alpha=0.5)
    ax.scatter(g["mean"], y, color=colors, s=22, zorder=3, edgecolor="white", lw=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xscale("symlog", linthresh=0.002)
    ax.set_yticks(y, [VAR_LABELS.get(v, v) for v in g.index], fontsize=7)
    ax.set_xlabel("Caída del PR-AUC al permutar la variable (escala simétrica logarítmica)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=BLUE, label="Se conserva"),
                       Patch(color=GRAY, label="Se descarta (media ≤ 0)")],
              fontsize=7, loc="lower right")
    return _save(fig, path)


# ------------------------------------------------------------------ modelos
def model_comparison(folds: pd.DataFrame, prevalence: float, path: Path) -> Path:
    """PR-AUC y F2 de cada modelo en los cinco pliegues externos (punto grande = media)."""
    order = folds.groupby("modelo")["pr_auc"].mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (metric, title) in zip(axes, [("pr_auc", "PR-AUC"), ("f2", "F2 (umbral calibrado)")],
                                   strict=True):
        for yi, mdl in enumerate(order):
            vals = folds.loc[folds.modelo == mdl, metric]
            c = GRAY if mdl in ("Mayoría", "Árbol sin poda") else BLUE
            ax.plot([vals.min(), vals.max()], [yi, yi], color=c, lw=2, solid_capstyle="round")
            ax.scatter(vals, [yi] * len(vals), s=10, color=c, alpha=0.5, lw=0)
            ax.scatter(vals.mean(), yi, s=34, color=c, zorder=3, edgecolor="white", lw=1)
        if metric == "pr_auc":
            ax.axvline(prevalence, color=GRAY, ls="--", lw=1)
            ax.text(prevalence, len(order) - 0.4, " azar", fontsize=7, color=INK2)
        ax.set_yticks(range(len(order)), order)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def pr_curves(oof: pd.DataFrame, models: list[str], prevalence: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for c, mdl in zip(SERIES + [GRAY], models, strict=False):
        d = oof[oof.modelo == mdl]
        p, r, _ = precision_recall_curve(d.y, d.score)
        ax.plot(r, p, color=c, lw=1.6, label=mdl)
    ax.axhline(prevalence, color=GRAY, ls="--", lw=1)
    ax.text(0.99, prevalence + 0.015, "azar", ha="right", fontsize=7, color=INK2)
    ax.set_xlabel("Recall (sensibilidad)")
    ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


def leakage(demo: pd.DataFrame, path: Path) -> Path:
    cols = ["f1_Retenido", "f1_Deserción", "f1_ponderado", "pr_auc"]
    mean = demo.groupby("protocolo")[cols].mean()
    order = ["Artículo (SMOTE antes de partir)", "Pipeline (SMOTE dentro del pliegue)"]
    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    w = 0.38
    for i, (proto, c) in enumerate(zip(order, [ORANGE, BLUE], strict=True)):
        vals = mean.loc[proto].values
        ax.bar(x + (i - 0.5) * (w + 0.02), vals, width=w, color=c, label=proto)
        for xi, v in zip(x, vals, strict=True):
            ax.text(xi + (i - 0.5) * (w + 0.02), v + 0.01, _num(v), ha="center", fontsize=7,
                    color=INK2)
    ax.set_xticks(x, ["F1 retenido", "F1 deserción", "F1 ponderado", "PR-AUC"])
    ax.set_ylabel("Media de 5 pliegues")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    return _save(fig, path)


CURVE_LABELS = {"clf__num_leaves": ("Hojas por árbol", 2),
                 "clf__svm__C": ("C (menor = más regularización)", 10)}


def validation_curve(curve: pd.DataFrame, path: Path) -> Path:
    """Curvas de validación: PR-AUC de entrenamiento y validación por modelo (media ± DE)."""
    models = list(dict.fromkeys(curve.modelo))
    fig, axes = plt.subplots(1, len(models), figsize=(3.7 * len(models), 2.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models, strict=True):
        c = curve[curve.modelo == m]
        param = c.parametro.iloc[0]
        g = c.groupby(["parte", "valor"])["pr_auc"].agg(["mean", "std"]).reset_index()
        for part, color in (("entrenamiento", GRAY), ("validación", BLUE)):
            d = g[g.parte == part]
            ax.plot(d.valor, d["mean"], color=color, lw=2, marker="o", ms=4,
                    label=part.capitalize())
            ax.fill_between(d.valor, d["mean"] - d["std"], d["mean"] + d["std"], color=color,
                            alpha=0.15, lw=0)
        label, base = CURVE_LABELS.get(param, (param, 10))
        vals = sorted(c.valor.unique())
        ax.set_xscale("log", base=base)
        ax.set_xticks(vals, [f"{v:g}".replace(".", ",") for v in vals])
        ax.minorticks_off()
        ax.set_xlabel(label)
        ax.set_title(m)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("PR-AUC")
    fig.tight_layout()
    return _save(fig, path)


def cost_vs_performance(summary: pd.DataFrame, cost: pd.DataFrame, path: Path) -> Path:
    """PR-AUC frente al tiempo de ajuste del modelo final (escala logarítmica)."""
    d = cost.set_index("modelo").join(summary[["pr_auc_mean", "pr_auc_std"]])
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for m, r in d.iterrows():
        base = m in ("Mayoría", "Árbol sin poda")
        c = GRAY if base else BLUE
        ax.errorbar(r["ajuste_s"], r["pr_auc_mean"], yerr=r["pr_auc_std"], fmt="o", ms=6,
                    color=c, ecolor=c, elinewidth=1, capsize=2, mec="white", mew=1)
        below = m in ("XGBoost", "Random Forest")  # evita que se encimen con sus vecinos
        ax.annotate(m, (r["ajuste_s"], r["pr_auc_mean"]), xytext=(5, -10 if below else 3),
                    textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de ajuste del modelo final (s, escala log.)")
    ax.set_ylabel("PR-AUC (media ± DE)")
    fig.tight_layout()
    return _save(fig, path)


# -------------------------------------------------------------- interpretación
def shap_importance(table: pd.DataFrame, path: Path, title: str = "", top: int = 12) -> Path:
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.24 * len(t) + 0.8))
    ax.barh([VAR_LABELS.get(v, v) for v in t.index], t["media"], xerr=t["sd"], color=BLUE,
            height=0.65, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    ax.set_xlabel("|SHAP| medio (log-odds)")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    return _save(fig, path)


def shap_beeswarm(last: dict, path: Path, top: int = 8) -> Path:
    """Beeswarm de las variables numéricas (el color solo tiene sentido en una escala ordenada)."""
    X = last["X"].apply(pd.to_numeric, errors="coerce").rename(columns=VAR_LABELS)
    keep = [i for i, c in enumerate(X.columns) if X[c].notna().any()]
    shap.summary_plot(last["shap"][:, keep], X.iloc[:, keep], max_display=top, show=False,
                      plot_size=(7.0, 3.2),
                      color_bar_label="Valor de la variable")
    fig = plt.gcf()
    ax, cbar = fig.axes[0], fig.axes[-1]
    ax.set_xlabel("Valor SHAP (impacto en log-odds de desertar)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    cbar.set_yticklabels(["Bajo", "Alto"], fontsize=8)
    cbar.set_ylabel("Valor de la variable", fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------------- EDA
# Rampa secuencial azul (magnitud u orden) y divergente azul <-> rojo con gris neutro al centro.
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
RED, NEUTRAL = "#e34948", "#f0efec"


def _label(v: str) -> str:
    return VAR_LABELS.get(v, v)


def eda_dropout_moment(table: pd.DataFrame, path: Path) -> Path:
    """Deserción por cohorte apilada según el momento en que ocurre (orden temporal -> rampa)."""
    moments = [c for c in table.columns if c not in ("n", "tasa_total")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    left = np.zeros(len(table))
    for m, c in zip(moments, ORDINAL, strict=True):
        vals = 100 * table[m].to_numpy()
        ax.barh(table.index, vals, left=left, color=c, height=0.62, label=m,
                edgecolor="white", linewidth=1)
        left += vals
    for yi, v in enumerate(left):
        ax.text(v + 0.15, yi, _num(v, 1), va="center", fontsize=7.5, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Deserción en el primer año (% de la cohorte)")
    ax.set_xlim(0, left.max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    return _save(fig, path)


def eda_missing_heatmap(table: pd.DataFrame, path: Path) -> Path:
    """% de datos ausentes por variable y cohorte."""
    t = table[(table > 1).any(axis=1)]
    t = t.loc[t.mean(axis=1).sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + SEQ)
    fig, ax = plt.subplots(figsize=(5.8, 0.3 * len(t) + 0.9))
    ax.imshow(t.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(t.to_numpy()):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if v > 55 else INK2)
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(len(t)), [_label(v) for v in t.index], fontsize=7.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Cohorte de ingreso")
    ax.set_title("% de registros sin dato (vacío o 'No information')", loc="left")
    return _save(fig, path)


def eda_missing_vs_target(table: pd.DataFrame, overall: float, path: Path) -> Path:
    """Tasa de deserción con dato vs. sin dato: ¿la ausencia es informativa?"""
    t = table.sort_values("diferencia_pp")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(t) + 0.9))
    ax.hlines(y, 100 * t.tasa_con_dato, 100 * t.tasa_sin_dato, color=GRID, lw=2)
    ax.scatter(100 * t.tasa_con_dato, y, s=30, color=GRAY, zorder=3, label="Con dato",
               edgecolor="white", lw=1)
    ax.scatter(100 * t.tasa_sin_dato, y, s=30, color=BLUE, zorder=3, label="Sin dato",
               edgecolor="white", lw=1)
    ax.axvline(100 * overall, color=GRAY, ls="--", lw=1)
    labels = [f"{_label(v)} ({p:.0f} %)" for v, p in zip(t.variable, t.pct_ausente, strict=True)]
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("Deserción en el primer año (%)  ·  entre paréntesis: % sin dato")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path)


def eda_numeric_profiles(profiles: dict[str, pd.DataFrame], overall: float, path: Path) -> Path:
    """Tasa de deserción por deciles de cada variable numérica (con IC de Wilson al 95 %)."""
    n, cols = len(profiles), 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.6, 2.0 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (var, g) in zip(axes, profiles.items(), strict=False):
        d = g[g.index != "Sin dato"]
        ax.fill_between(d["centro"], 100 * d["ic_inf"], 100 * d["ic_sup"], color=BLUE, alpha=0.15,
                        lw=0)
        ax.plot(d["centro"], 100 * d["tasa"], color=BLUE, lw=2, marker="o", ms=3.5)
        ax.axhline(100 * overall, color=GRAY, ls="--", lw=1)
        ax.set_title(textwrap.fill(_label(var), 24), fontsize=8.5)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[::cols]:
        ax.set_ylabel("Deserción (%)")
    fig.tight_layout()
    return _save(fig, path)


def eda_categorical_profiles(profiles: dict[str, pd.DataFrame], overall: float, path: Path) -> Path:
    """Tasa de deserción por categoría (punto + IC 95 %), una variable por panel."""
    names = list(profiles)
    n, cols = len(names), 2
    rows = int(np.ceil(n / cols))
    heights = [max(len(profiles[v]) for v in names[r * cols:(r + 1) * cols]) for r in range(rows)]
    fig, axes = plt.subplots(rows, cols, figsize=(7.6, 0.2 * sum(heights) + 0.7 * rows),
                             gridspec_kw={"height_ratios": heights}, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (var, g) in zip(axes, profiles.items(), strict=False):
        y = np.arange(len(g))
        ax.hlines(y, 100 * g["ic_inf"], 100 * g["ic_sup"], color=BLUE, lw=1.5, alpha=0.5)
        ax.scatter(100 * g["tasa"], y, s=22, color=BLUE, zorder=3, edgecolor="white", lw=1)
        ax.axvline(100 * overall, color=GRAY, ls="--", lw=1)
        labels = [f"{str(c)[:28]} (n={int(k):,})".replace(",", ".")
                  for c, k in zip(g.index, g["n"], strict=True)]
        ax.set_yticks(y, labels, fontsize=6.8)
        ax.set_title(_label(var), fontsize=8.5, loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[-cols:]:
        ax.set_xlabel("Deserción (%)")
    fig.tight_layout()
    return _save(fig, path)


def eda_association(tables: dict[str, pd.DataFrame], highlight: list[str], path: Path) -> Path:
    """Fuerza de la asociación univariada (2·|AUC − 0,5|) de cada variable con la deserción."""
    height = 0.17 * max(len(t) for t in tables.values()) + 1
    fig, axes = plt.subplots(1, len(tables), figsize=(7.6, height))
    axes = np.atleast_1d(axes)
    for ax, (title, t) in zip(axes, tables.items(), strict=True):
        t = t.iloc[::-1]
        colors = [ORANGE if v in highlight else BLUE for v in t.variable]
        ax.barh([_label(v) for v in t.variable], t["fuerza_auc"], color=colors, height=0.7)
        ax.set_title(title)
        ax.set_xlabel("2·|AUC univariada − 0,5|")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=6.8)
    if highlight:
        axes[-1].legend(handles=[Patch(color=BLUE, label="Información de ingreso"),
                                 Patch(color=ORANGE, label="Desempeño del 1.er semestre")],
                        fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def eda_correlation(corr: pd.DataFrame, path: Path) -> Path:
    """Matriz de correlación de Spearman entre variables numéricas (divergente, 0 = gris)."""
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    # Triángulo inferior sin la diagonal (ρ = 1 no aporta): filas 2..n contra columnas 1..n-1.
    full = np.where(np.triu(np.ones(corr.shape, dtype=bool)), np.nan, corr.to_numpy())
    data = full[1:, :-1]
    ylabels, xlabels = [_label(v) for v in corr.index[1:]], [_label(v) for v in corr.columns[:-1]]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(data):
        if not np.isnan(v):
            ax.text(j, i, _num(v), ha="center", va="center", fontsize=5.8,
                    color="white" if abs(v) > 0.6 else INK2)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("ρ de Spearman", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    return _save(fig, path)
