"""Figuras del informe. Paleta categórica validada (azul, naranja, aqua) y gris para referencias."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

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
def target_by_cohort(df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.4, 1]})
    rate = df.groupby("generation")["desercion"].mean() * 100
    ax = axes[0]
    ax.bar(rate.index, rate.values, color=BLUE, width=0.6)
    for x, v in zip(rate.index, rate.values, strict=True):
        ax.text(x, v + 0.15, _num(v, 1), ha="center", va="bottom", fontsize=8, color=INK2)
    mean = df["desercion"].mean() * 100
    ax.axhline(mean, color=GRAY, ls="--", lw=1)
    ax.set_ylim(0, 12.2)
    ax.text(len(rate) - 0.6, 11.6, f"– – promedio {_num(mean, 1)}", ha="right", fontsize=7, color=INK2)
    ax.set_ylabel("Deserción en el primer año (%)")
    ax.set_title("(a) Tasa de deserción por cohorte")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    labels = {1: "Durante 1er sem.", 2: "No se inscribe 2º", 3: "Durante 2º sem.",
              4: "No se inscribe 3º"}
    counts = df.loc[df["dropout.semester"] > 0, "dropout.semester"].value_counts().sort_index()
    ax.barh([labels[i] for i in counts.index], counts.values, color=[BLUE, BLUE, ORANGE, ORANGE],
            height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + 60, i, f"{v:,}".replace(",", "."), va="center", fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("Desertores")
    ax.set_title("(b) Momento de la deserción")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.25)
    fig.tight_layout()
    return _save(fig, path)


def data_quality(q: pd.DataFrame, used: list[str], path: Path) -> Path:
    q = q[q.variable.isin(used)].copy()
    q["ausente"] = q["pct_nan"] + q["pct_no_information"]
    q = q[q["ausente"] > 1].sort_values("ausente")
    fig, ax = plt.subplots(figsize=(6.0, 0.22 * len(q) + 0.8))
    ax.barh(q.variable, q["pct_nan"], color=BLUE, height=0.65, label="Vacío / no aplica (NaN)")
    ax.barh(q.variable, q["pct_no_information"], left=q["pct_nan"], color=ORANGE, height=0.65,
            label="Codificado 'No information'", edgecolor="white", linewidth=1)
    ax.set_xlabel("% de registros de pregrado")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


# ------------------------------------------------------------------ modelos
def model_comparison(folds: dict[str, pd.DataFrame], prevalence: dict[str, float], path: Path) -> Path:
    scen = list(folds)
    fig, axes = plt.subplots(len(scen), 2, figsize=(7.2, 2.5 * len(scen)), sharey="row")
    axes = np.atleast_2d(axes)
    for i, s in enumerate(scen):
        f = folds[s]
        order = f.groupby("modelo")["pr_auc"].mean().sort_values().index
        for j, (metric, title) in enumerate([("pr_auc", "PR-AUC"), ("f2", "F2 (umbral calibrado)")]):
            ax = axes[i, j]
            for yi, mdl in enumerate(order):
                vals = f.loc[f.modelo == mdl, metric]
                base = mdl in ("Mayoría", "Árbol sin poda")
                c = GRAY if base else BLUE
                ax.plot([vals.min(), vals.max()], [yi, yi], color=c, lw=2, solid_capstyle="round")
                ax.scatter(vals, [yi] * len(vals), s=10, color=c, alpha=0.5, lw=0)
                ax.scatter(vals.mean(), yi, s=34, color=c, zorder=3, edgecolor="white", lw=1)
            if metric == "pr_auc":
                ax.axvline(prevalence[s], color=GRAY, ls="--", lw=1)
            ax.set_yticks(range(len(order)), order)
            ax.set_title(f"Escenario {s} — {title}")
            ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def pr_curves(oof: dict[str, pd.DataFrame], models: dict[str, list[str]], prevalence: dict[str, float],
              path: Path) -> Path:
    fig, axes = plt.subplots(1, len(oof), figsize=(7.2, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (s, o) in zip(axes, oof.items(), strict=True):
        for c, mdl in zip(SERIES + [GRAY], models[s], strict=False):
            d = o[o.modelo == mdl]
            p, r, _ = precision_recall_curve(d.y, d.score)
            ax.plot(r, p, color=c, lw=1.6, label=mdl)
        ax.axhline(prevalence[s], color=GRAY, ls="--", lw=1)
        ax.text(0.99, prevalence[s] + 0.015, "azar", ha="right", fontsize=7, color=INK2)
        ax.set_xlabel("Recall (sensibilidad)")
        ax.set_title(f"Escenario {s}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Precisión")
    fig.tight_layout()
    return _save(fig, path)


def leakage(demo: pd.DataFrame, path: Path) -> Path:
    cols = [c for c in demo.columns if c.startswith("f1_") and c not in ("f1_macro",)]
    cols = [c for c in cols if c != "f1_ponderado"] + ["f1_ponderado"]
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
    labels = [c.removeprefix("f1_").replace("_", " ").capitalize() for c in cols]
    ax.set_xticks(x, [lbl.replace(" ", "\n", 1) if len(lbl) > 12 else lbl for lbl in labels])
    ax.set_ylabel("F1 (media 5 pliegues)")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    return _save(fig, path)


def ablation(abl: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(abl), figsize=(7.2, 2.6))
    axes = np.atleast_1d(axes)
    for ax, (s, a) in zip(axes, abl.items(), strict=True):
        g = a.groupby("conjunto", sort=False)["pr_auc"].agg(["mean", "std"])
        y = np.arange(len(g))
        ax.barh(y, g["mean"], xerr=g["std"], color=BLUE, height=0.6,
                error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        for yi, v, sd in zip(y, g["mean"], g["std"], strict=True):
            ax.text(v + sd + 0.008, yi, _num(v), va="center", fontsize=7, color=INK2)
        ax.set_yticks(y, g.index)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.3)
        ax.set_xlabel("PR-AUC (LightGBM sin SMOTE)")
        ax.set_title(f"Escenario {s}")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def validation_curve(curves: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(curves), figsize=(7.2, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (s, c) in zip(axes, curves.items(), strict=True):
        g = c.groupby(["parte", "num_leaves"])["pr_auc"].agg(["mean", "std"]).reset_index()
        for part, color in (("entrenamiento", GRAY), ("validación", BLUE)):
            d = g[g.parte == part]
            ax.plot(d.num_leaves, d["mean"], color=color, lw=2, marker="o", ms=4, label=part.capitalize())
            ax.fill_between(d.num_leaves, d["mean"] - d["std"], d["mean"] + d["std"], color=color,
                            alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(c.num_leaves.unique()), [str(v) for v in sorted(c.num_leaves.unique())])
        ax.minorticks_off()
        ax.set_xlabel("Hojas por árbol (LightGBM)")
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"Escenario {s}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


# -------------------------------------------------------------- interpretación
def shap_importance(tables: dict[str, pd.DataFrame], path: Path, titles: dict[str, str] | None = None,
                    top: int = 12) -> Path:
    fig, axes = plt.subplots(1, len(tables), figsize=(7.2, 3.3))
    axes = np.atleast_1d(axes)
    for ax, (s, t) in zip(axes, tables.items(), strict=True):
        t = t.head(top).iloc[::-1]
        ax.barh(t.index, t["media"], xerr=t["sd"], color=BLUE, height=0.65,
                error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        ax.set_xlabel("|SHAP| medio (log-odds)")
        ax.set_title((titles or {}).get(s, f"Escenario {s}"))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    return _save(fig, path)


def shap_beeswarm(last: dict, path: Path, top: int = 8) -> Path:
    """Beeswarm de las variables numéricas (el color solo tiene sentido en una escala ordenada)."""
    X = last["X"].apply(pd.to_numeric, errors="coerce")
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
