"""Tablas LaTeX del informe generadas directamente desde los resultados (sin transcribir cifras)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ORDER = ["Mayoría", "Árbol sin poda", "Regresión logística", "Random Forest",
         "SVM (RBF-Nystroem)", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost",
         "CatBoost nativo"]
LABELS = {"CatBoost nativo": "CatBoost nativo$^{\\dagger}$", "SVM (RBF-Nystroem)": "SVM (RBF)"}


def _fmt(mean: float, sd: float | None = None, d: int = 3) -> str:
    txt = f"{mean:.{d}f}".replace(".", "{,}")
    if sd is not None:
        txt += f"\\,\\scriptsize$\\pm${sd:.{d}f}".replace(".", "{,}")
    return txt


def results_table(summaries: dict[str, pd.DataFrame], path: Path) -> Path:
    metrics = [("pr_auc", True), ("roc_auc", False), ("f2", True), ("recall", False)]
    bold = {"pr_auc", "roc_auc", "f2"}  # el recall máximo no es "mejor" si la precisión cae
    best = {s: {m: summaries[s][f"{m}_mean"].drop(["Mayoría", "Árbol sin poda"], errors="ignore").max()
                for m, _ in metrics} for s in summaries}
    lines = []
    for model in ORDER:
        cells = [LABELS.get(model, model)]
        for s, summ in summaries.items():
            if model not in summ.index:
                cells += ["--"] * len(metrics)
                continue
            for m, with_sd in metrics:
                v = summ.loc[model, f"{m}_mean"]
                cell = _fmt(v, summ.loc[model, f"{m}_std"] if with_sd else None)
                if m in bold and abs(v - best[s][m]) < 1e-12:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
        if model == "Árbol sin poda":
            lines.append("\\midrule")
    head = " & ".join(["Modelo"] + ["PR-AUC", "ROC-AUC", "F2", "Recall"] * len(summaries))
    groups = " & ".join([""] + [f"\\multicolumn{{4}}{{c}}{{Escenario {s}}}" for s in summaries])
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(summaries)))
    body = "\n".join(lines)
    tex = ("\\setlength{\\tabcolsep}{3.5pt}\n"
           f"\\begin{{tabular}}{{l{'rrrr' * len(summaries)}}}\n\\toprule\n{groups} \\\\\n{cmid}\n"
           f"{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")
    path.write_text(tex, encoding="utf-8")
    return path
