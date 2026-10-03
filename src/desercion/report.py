"""Tablas LaTeX del informe generadas directamente desde los resultados (sin transcribir cifras)."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

ORDER = ["Mayoría", "Árbol sin poda", "Regresión logística", "Random Forest",
         "SVM (RBF-Nystroem)", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]
LABELS = {"SVM (RBF-Nystroem)": "SVM (kernel RBF)"}

# Nombre legible de cada hiperparámetro de las rejillas.
PARAM_NAMES = {
    "clf__C": "$C$",
    "clf__min_samples_leaf": "mín. registros por hoja",
    "clf__max_features": "variables por división",
    "clf__kernel__gamma": r"$\gamma$",
    "clf__svm__C": "$C$",
    "clf__learning_rate": "tasa de aprendizaje",
    "clf__max_leaf_nodes": "máx. hojas",
    "clf__l2_regularization": "regularización L2",
    "clf__max_depth": "profundidad máx.",
    "clf__min_child_weight": "peso mín. por hoja",
    "clf__num_leaves": "hojas",
    "clf__min_child_samples": "mín. registros por hoja",
    "clf__depth": "profundidad",
    "clf__l2_leaf_reg": "regularización L2",
    "clf__class_weights": "pesos de clase",
    "sampler": "balanceo",
}


def _dec(v) -> str:
    """Valor de un hiperparámetro en texto (los objetos de balanceo, por su nombre)."""
    if v is None or v in ("passthrough", "none"):
        return "ninguno"
    if isinstance(v, str):
        return {"Balanced": "balanceados", "sqrt": r"$\sqrt{p}$"}.get(v, v)
    if not isinstance(v, (int, float)):
        return type(v).__name__
    return f"{v:g}".replace(".", "{,}")


def _fmt(mean: float, sd: float | None = None, d: int = 3) -> str:
    txt = f"{mean:.{d}f}".replace(".", "{,}")
    if sd is not None:
        txt += rf"\,\scriptsize$\pm${sd:.{d}f}".replace(".", "{,}")
    return txt


def _tabular(spec: str, header: str, lines: list[str]) -> str:
    return ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def results_table(summary: pd.DataFrame, path: Path) -> Path:
    """PR-AUC y ROC-AUC (media ± desviación), recall, precisión y tasa de alerta por modelo."""
    best = summary["pr_auc_mean"].drop(["Mayoría", "Árbol sin poda"], errors="ignore").max()
    lines = []
    for model in ORDER:
        r = summary.loc[model]
        pr = _fmt(r["pr_auc_mean"], r["pr_auc_std"])
        if abs(r["pr_auc_mean"] - best) < 1e-12:
            pr = f"\\textbf{{{pr}}}"
        cells = [LABELS.get(model, model), pr, _fmt(r["roc_auc_mean"], r["roc_auc_std"]),
                 _fmt(r["recall_mean"], d=2), _fmt(r["precision_mean"], d=2),
                 _fmt(r["tasa_alerta_mean"], d=2)]
        lines.append(" & ".join(cells) + " \\\\")
        if model == "Árbol sin poda":
            lines.append("\\midrule")
    header = "Modelo & PR-AUC & ROC-AUC & Recall & Precisión & Alertas"
    path.write_text(_tabular("lrrrrr", header, lines), encoding="utf-8")
    return path


def hyperparams_table(folds: pd.DataFrame, grids: dict[str, dict], path: Path) -> Path:
    """Rejilla explorada y combinación elegida con más frecuencia en los pliegues externos."""
    lines = []
    for model in ORDER:
        grid = grids.get(model)
        if not grid:
            continue
        explored = "; ".join(f"{PARAM_NAMES[p]}: {', '.join(_dec(v) for v in vals)}"
                             for p, vals in grid.items())
        params = folds.loc[folds.modelo == model, "params"]
        top = params.mode().iloc[0]
        mode = ast.literal_eval(top)
        chosen = ", ".join(_dec(mode[p]) for p in grid)
        n = int((params == top).sum())
        lines.append(f"{LABELS.get(model, model)} & {explored} & {chosen} ({n}/5) \\\\")
    spec = (r">{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{8.2cm}"
            r">{\raggedright\arraybackslash}p{3.6cm}")
    header = "Modelo & Valores explorados & Elegido (rondas)"
    path.write_text(_tabular(spec, header, lines), encoding="utf-8")
    return path


def leakage_table(demo: pd.DataFrame, path: Path) -> Path:
    """Mismo LightGBM con el protocolo del artículo y con el Pipeline sin fuga."""
    g = demo.groupby("protocolo")[["f1_Deserción", "pr_auc"]].mean()
    names = {"Artículo (SMOTE antes de partir)": "SMOTE antes de separar (artículo)",
             "Pipeline (SMOTE dentro del pliegue)": "SMOTE solo en entrenamiento (este trabajo)"}
    lines = [f"{names[p]} & {_fmt(r['f1_Deserción'], d=2)} & {_fmt(r['pr_auc'], d=2)} \\\\"
             for p, r in g.loc[list(names)].iterrows()]
    header = "Protocolo & F1 deserción (umbral 0,5) & PR-AUC"
    path.write_text(_tabular("lcc", header, lines), encoding="utf-8")
    return path


def cost_table(cost: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    """Costo computacional frente a desempeño, por modelo."""
    c = cost.set_index("modelo")
    lines = []
    for model in ORDER:
        r = c.loc[model]
        lines.append(" & ".join([
            LABELS.get(model, model), _fmt(summary.loc[model, "pr_auc_mean"]),
            str(int(r["combinaciones"])), _fmt(r["busqueda_min"], d=1), _fmt(r["ajuste_s"], d=2),
            _fmt(r["prediccion_ms_por_1000"], d=1), _fmt(r["tamano_mb"], d=2)]) + " \\\\")
    header = ("Modelo & PR-AUC & Combinaciones & Búsqueda (min) & Ajuste (s) & "
              "Predicción (ms/1.000) & Tamaño (MB)")
    path.write_text(_tabular("lrrrrrr", header, lines), encoding="utf-8")
    return path
