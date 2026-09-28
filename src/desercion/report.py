"""Tablas LaTeX del informe generadas directamente desde los resultados (sin transcribir cifras)."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

ORDER = ["Mayoría", "Árbol sin poda", "Regresión logística", "Random Forest",
         "SVM (RBF-Nystroem)", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost",
         "CatBoost nativo"]
LABELS = {"CatBoost nativo": "CatBoost nativo$^{\\dagger}$", "SVM (RBF-Nystroem)": "SVM (kernel RBF)"}

# Nombre legible de cada hiperparámetro de las rejillas.
PARAM_NAMES = {
    "clf__C": "$C$",
    "clf__max_depth": "profundidad máx.",
    "clf__min_samples_leaf": "mín. registros por hoja",
    "clf__kernel__gamma": "$\\gamma$",
    "clf__svm__C": "$C$",
    "clf__learning_rate": "tasa de aprendizaje",
    "clf__max_leaf_nodes": "máx. hojas",
    "clf__num_leaves": "hojas",
    "clf__depth": "profundidad",
}


def _dec(v) -> str:
    if v is None:
        return "sin límite"
    return f"{v:g}".replace(".", "{,}")


def _fmt(mean: float, sd: float | None = None, d: int = 3) -> str:
    txt = f"{mean:.{d}f}".replace(".", "{,}")
    if sd is not None:
        txt += f"\\,\\scriptsize$\\pm${sd:.{d}f}".replace(".", "{,}")
    return txt


def results_table(summaries: dict[str, pd.DataFrame], path: Path) -> Path:
    """PR-AUC (media ± desviación), recall y precisión por modelo y escenario."""
    metrics = [("pr_auc", True), ("recall", False), ("precision", False)]
    best = {s: summaries[s]["pr_auc_mean"].drop(["Mayoría", "Árbol sin poda"], errors="ignore").max()
            for s in summaries}
    lines = []
    for model in ORDER:
        cells = [LABELS.get(model, model)]
        for s, summ in summaries.items():
            for m, with_sd in metrics:
                v = summ.loc[model, f"{m}_mean"]
                cell = _fmt(v, summ.loc[model, f"{m}_std"] if with_sd else None, 2 if m != "pr_auc" else 3)
                if m == "pr_auc" and abs(v - best[s]) < 1e-12:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
        if model == "Árbol sin poda":
            lines.append("\\midrule")
    k = len(metrics)
    head = " & ".join(["Modelo"] + ["PR-AUC", "Recall", "Precisión"] * len(summaries))
    groups = " & ".join([""] + [f"\\multicolumn{{{k}}}{{c}}{{Escenario {s}}}" for s in summaries])
    cmid = " ".join(f"\\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(len(summaries)))
    tex = (f"\\begin{{tabular}}{{l{'rrr' * len(summaries)}}}\n\\toprule\n{groups} \\\\\n{cmid}\n"
           f"{head} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    path.write_text(tex, encoding="utf-8")
    return path


def hyperparams_table(folds: dict[str, pd.DataFrame], grids: dict[str, dict], path: Path) -> Path:
    """Rejilla explorada y valor elegido con más frecuencia en los pliegues externos."""
    lines = []
    for model in ORDER:
        grid = grids.get(model)
        if not grid:
            continue
        explored = "; ".join(f"{PARAM_NAMES[p]}: {', '.join(_dec(v) for v in vals)}"
                             for p, vals in grid.items())
        chosen = []
        for f in folds.values():
            mode = ast.literal_eval(f.loc[f.modelo == model, "params"].mode().iloc[0])
            chosen.append(", ".join(_dec(mode[p]) for p in grid))
        lines.append(" & ".join([LABELS.get(model, model), explored, *chosen]) + " \\\\")
    cols = " & ".join(["Modelo", "Valores explorados"] + [f"Elegido {s}" for s in folds])
    tex = ("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.9cm}"
           ">{\\raggedright\\arraybackslash}p{7.3cm}" + "c" * len(folds) + "}\n\\toprule\n"
           f"{cols} \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    path.write_text(tex, encoding="utf-8")
    return path


def leakage_table(demo: pd.DataFrame, path: Path) -> Path:
    """Mismo LightGBM con el protocolo del artículo y con el Pipeline sin fuga."""
    g = demo.groupby("protocolo")[["f1_Deserción", "pr_auc"]].mean()
    names = {"Artículo (SMOTE antes de partir)": "SMOTE antes de separar (artículo)",
             "Pipeline (SMOTE dentro del pliegue)": "SMOTE solo en entrenamiento (este trabajo)"}
    lines = [f"{names[p]} & {_fmt(r['f1_Deserción'], d=2)} & {_fmt(r['pr_auc'], d=2)} \\\\"
             for p, r in g.loc[list(names)].iterrows()]
    tex = ("\\begin{tabular}{lcc}\n\\toprule\n"
           "Protocolo & F1 deserción (umbral 0,5) & PR-AUC \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    path.write_text(tex, encoding="utf-8")
    return path
