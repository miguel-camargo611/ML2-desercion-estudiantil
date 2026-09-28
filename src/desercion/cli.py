"""Punto de entrada del flujo reproducible.

Uso (desde la raíz del repositorio, con el entorno activado):

    python -m desercion.cli prepare       # limpieza, partición sellada, calidad de datos
    python -m desercion.cli train         # CV anidada de todos los modelos (escenarios A y B)
    python -m desercion.cli compare       # pruebas estadísticas entre modelos
    python -m desercion.cli experiments   # SMOTE/ADASYN, fuga, Isolation Forest, ablación, multiclase, VIF
    python -m desercion.cli interpret     # SHAP por pliegue y análisis de errores
    python -m desercion.cli figures       # figuras del informe
    python -m desercion.cli report        # copia las figuras a Segunda_entrega/figuras
    python -m desercion.cli all           # todo lo anterior, en orden
"""

from __future__ import annotations

import argparse
import ast
import logging
import shutil
import warnings

import pandas as pd
from sklearn.base import clone

from desercion import config, data, evaluate, experiments, interpret, plots, report
from desercion.features import EXCLUDED, feature_lists
from desercion.models import get_specs

log = logging.getLogger("desercion")
TREE_MODELS = ["LightGBM", "CatBoost", "CatBoost nativo", "XGBoost", "Gradient Boosting",
               "Random Forest"]


def _tables():
    return config.path("tables_dir")


def _figures():
    return config.path("figures_dir")


def _dev(scenario: str, sample: float | None = None) -> pd.DataFrame:
    df = data.load_clean()
    split = data.get_split(df)
    dev = data.scenario_frame(df, scenario, split, "dev")
    if sample:
        dev = dev.groupby(data.TARGET, group_keys=False).sample(frac=sample,
                                                               random_state=config.seed())
    return dev


# ----------------------------------------------------------------------- pasos
def cmd_prepare(args) -> None:
    df = data.load_clean()
    split = data.get_split(df, recompute=args.recompute_split)
    rows = []
    for s in data.SCENARIOS:
        for part in ("dev", "test"):
            f = data.scenario_frame(df, s, split, part)
            rows.append({"escenario": s, "parte": part, "registros": len(f),
                         "estudiantes": f[data.GROUP].nunique(), "desertores": int(f[data.TARGET].sum()),
                         "tasa_desercion": f[data.TARGET].mean()})
    resumen = pd.DataFrame(rows)
    resumen.to_csv(_tables() / "datos_resumen.csv", index=False)
    log.info("\n%s", resumen.to_string(index=False))
    grupos_repetidos = df.loc[split == "dev", data.GROUP].isin(df.loc[split == "test", data.GROUP]).sum()
    log.info("Estudiantes presentes en desarrollo y prueba a la vez: %d", grupos_repetidos)

    q = data.quality_report(df)
    q.to_csv(_tables() / "calidad_datos.csv", index=False)
    pd.Series(EXCLUDED, name="motivo").rename_axis("variable").to_csv(_tables() / "variables_excluidas.csv")
    num, cat = feature_lists("B")
    plots.target_by_cohort(df, _figures() / "fig01_desercion_cohorte.png")
    plots.data_quality(q, num + cat, _figures() / "fig02_calidad_datos.png")


def cmd_train(args) -> None:
    specs = get_specs()
    names = args.models or list(specs)
    for s in args.scenarios:
        dev = _dev(s, args.sample)
        num, cat = feature_lists(s)
        X, y, g = dev[num + cat], dev[data.TARGET], dev[data.GROUP]
        log.info("Escenario %s: %d registros, %.2f%% desertores", s, len(dev), 100 * y.mean())
        f_path, o_path = _tables() / f"cv_pliegues_{s}.csv", config.path("processed_dir") / f"oof_{s}.parquet"
        folds = pd.read_csv(f_path) if f_path.exists() else pd.DataFrame()
        oofs = pd.read_parquet(o_path) if o_path.exists() else pd.DataFrame()
        for name in names:
            fr, oo = evaluate.nested_cv(specs[name], X, y, g, num, cat)
            if not folds.empty:
                folds = folds[folds.modelo != name]
                oofs = oofs[oofs.modelo != name]
            folds = pd.concat([folds, fr], ignore_index=True)
            oofs = pd.concat([oofs, oo], ignore_index=True)
            folds.to_csv(f_path, index=False)
            oofs.to_parquet(o_path)
        evaluate.summarize(folds).to_csv(_tables() / f"cv_resumen_{s}.csv")
        log.info("\n%s", evaluate.summarize(folds)[["pr_auc_mean", "pr_auc_std", "f2_mean",
                                                     "recall_mean", "precision_mean"]].round(3))


def cmd_compare(args) -> None:
    cfg = config.load_config()
    k = cfg["cv"]["outer_folds"]
    rows = []
    for s in args.scenarios:
        folds = pd.read_csv(_tables() / f"cv_pliegues_{s}.csv")
        ranking = folds.groupby("modelo")["pr_auc"].mean().sort_values(ascending=False).index
        best, second = ranking[0], ranking[1]
        n = len(_dev(s))
        n_test = n // k
        pairs = [(best, second), (best, "Árbol sin poda"), (best, "Regresión logística"),
                 (best, "LightGBM"), ("LightGBM", "CatBoost"), ("LightGBM", "Regresión logística")]
        for a, b in dict.fromkeys(pairs):
            if a == b or b not in set(folds.modelo) or a not in set(folds.modelo):
                continue
            for metric in ("pr_auc", "f2"):
                r = evaluate.compare_models(folds, a, b, metric, n - n_test, n_test)
                rows.append({"escenario": s, **r})
    out = pd.DataFrame(rows)
    out.to_csv(_tables() / "pruebas_estadisticas.csv", index=False)
    log.info("\n%s", out.round(4).to_string(index=False))


EXPERIMENTS = ["vif", "ablacion", "isolation_forest", "curva_validacion", "balanceo_lgbm",
               "balanceo_arbol", "fuga", "multiclase"]


def cmd_experiments(args) -> None:
    t = _tables()
    todo = set(args.experiments or EXPERIMENTS)
    for s in args.scenarios:
        dev = _dev(s, args.sample)
        if "vif" in todo:
            experiments.vif(dev, s).to_csv(t / f"vif_{s}.csv", index=False)
        if "ablacion" in todo:
            experiments.ablation(dev, s).to_csv(t / f"ablacion_{s}.csv", index=False)
        if "isolation_forest" in todo:
            experiments.isolation_forest(dev, s).to_csv(t / f"isolation_forest_{s}.csv", index=False)
        if "curva_validacion" in todo:
            experiments.validation_curve_lgbm(dev, s).to_csv(t / f"curva_validacion_{s}.csv", index=False)
        if "balanceo_lgbm" in todo:
            experiments.balancing_best_model(dev, s).to_csv(t / f"balanceo_lgbm_{s}.csv", index=False)
        if s == "A":
            if "balanceo_arbol" in todo:
                experiments.resampling_comparison(dev, s).to_csv(t / "balanceo_A.csv", index=False)
            if "fuga" in todo:
                experiments.leakage_demo(dev, s).to_csv(t / "fuga_smote_A.csv", index=False)
                experiments.leakage_demo(dev, s, multiclass=True).to_csv(
                    t / "fuga_smote_multiclase_A.csv", index=False)
            if "multiclase" in todo:
                experiments.multiclass(dev, s).to_csv(t / "multiclase_A.csv", index=False)
        log.info("Experimentos del escenario %s listos", s)


def _best_tree_spec(s: str, forced: str | None):
    folds = pd.read_csv(_tables() / f"cv_pliegues_{s}.csv")
    ranking = folds.groupby("modelo")["pr_auc"].mean().sort_values(ascending=False)
    name = forced or next(m for m in ranking.index if m in TREE_MODELS)
    params = folds.loc[folds.modelo == name, "params"].mode().iloc[0]
    spec = get_specs()[name]
    est = clone(spec.estimator).set_params(**{k.removeprefix("clf__"): v
                                               for k, v in ast.literal_eval(params).items()})
    spec.estimator = est
    return spec, params


def cmd_interpret(args) -> None:
    t = _tables()
    for s in args.scenarios:
        dev = _dev(s, args.sample)
        spec, params = _best_tree_spec(s, args.model or config.load_config().get("interpret_model"))
        log.info("Interpretando %s en escenario %s con %s", spec.name, s, params)
        table, last = interpret.shap_by_fold(dev, s, spec)
        table.to_csv(t / f"shap_{s}.csv")
        pd.to_pickle(last, config.path("processed_dir") / f"shap_ultimo_pliegue_{s}.pkl")

        oof_all = pd.read_parquet(config.path("processed_dir") / f"oof_{s}.parquet")
        evaluate.capture_at_top(oof_all).to_csv(t / f"captura_topk_{s}.csv", index=False)
        oof = oof_all[oof_all.modelo == spec.name]
        segs = ["school", "region", "first.generation", "scholarship.type", "foreign"]
        interpret.error_analysis(oof, dev, segs).to_csv(t / f"errores_{s}.csv", index=False)
        cm = pd.crosstab(oof.y.map({0: "Retenido", 1: "Deserción"}),
                         oof.pred.map({0: "Pred. retenido", 1: "Pred. deserción"}))
        cm.to_csv(t / f"confusion_{s}.csv")
        log.info("Matriz de confusión (%s, %s):\n%s", s, spec.name, cm)


def cmd_figures(args) -> None:
    t, f = _tables(), _figures()
    scen = [s for s in args.scenarios if (t / f"cv_pliegues_{s}.csv").exists()]
    if scen:
        resumen = pd.read_csv(t / "datos_resumen.csv")
        prev = {s: resumen.query("escenario == @s and parte == 'dev'")["tasa_desercion"].iloc[0]
                for s in scen}
        folds = {s: pd.read_csv(t / f"cv_pliegues_{s}.csv") for s in scen}
        plots.model_comparison(folds, prev, f / "fig03_comparacion_modelos.png")
        oof = {s: pd.read_parquet(config.path("processed_dir") / f"oof_{s}.parquet") for s in scen}
        top = {}
        for s in scen:
            rk = folds[s].groupby("modelo")["pr_auc"].mean().sort_values(ascending=False).index
            top[s] = [m for m in rk if m not in ("Árbol sin poda", "Mayoría")][:3] + ["Árbol sin poda"]
        plots.pr_curves(oof, top, prev, f / "fig04_curvas_pr.png")
    if (t / "fuga_smote_multiclase_A.csv").exists():
        plots.leakage(pd.read_csv(t / "fuga_smote_multiclase_A.csv"), f / "fig05_fuga_smote.png")
    abl = {s: pd.read_csv(t / f"ablacion_{s}.csv") for s in args.scenarios
           if (t / f"ablacion_{s}.csv").exists()}
    if abl:
        plots.ablation(abl, f / "fig06_ablacion.png")
    vc = {s: pd.read_csv(t / f"curva_validacion_{s}.csv") for s in args.scenarios
          if (t / f"curva_validacion_{s}.csv").exists()}
    if vc:
        plots.validation_curve(vc, f / "fig09_curva_validacion.png")
    shp = {s: pd.read_csv(t / f"shap_{s}.csv", index_col=0) for s in args.scenarios
           if (t / f"shap_{s}.csv").exists()}
    if shp:
        name = args.model or config.load_config().get("interpret_model")
        titles = {s: f"Escenario {s} — {_best_tree_spec(s, name)[0].name}" for s in shp}
        plots.shap_importance(shp, f / "fig07_shap_importancia.png", titles)
        last = config.path("processed_dir") / "shap_ultimo_pliegue_B.pkl"
        if last.exists():
            plots.shap_beeswarm(pd.read_pickle(last), f / "fig08_shap_beeswarm_B.png")


def cmd_report(args) -> None:
    dst = config.path("report_figures_dir")
    for fig in _figures().glob("*.png"):
        shutil.copy2(fig, dst / fig.name)
    summaries = {s: pd.read_csv(_tables() / f"cv_resumen_{s}.csv", index_col=0) for s in ("A", "B")
                 if (_tables() / f"cv_resumen_{s}.csv").exists()}
    if summaries:
        tex = report.results_table(summaries, _tables() / "tabla_resultados.tex")
        shutil.copy2(tex, dst.parent / "tabla_resultados.tex")
    log.info("Figuras y tablas copiadas a %s", dst.parent)


def cmd_all(args) -> None:
    for step in (cmd_prepare, cmd_train, cmd_compare, cmd_experiments, cmd_interpret, cmd_figures,
                 cmd_report):
        step(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="desercion", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "train", "compare", "experiments", "interpret",
                                         "figures", "report", "all"])
    parser.add_argument("--scenarios", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--models", nargs="+", default=None, help="Subconjunto de modelos a entrenar.")
    parser.add_argument("--experiments", nargs="+", default=None, choices=EXPERIMENTS,
                        help="Subconjunto de experimentos (por defecto, todos).")
    parser.add_argument("--model", default=None, help="Modelo a interpretar (por defecto, el mejor de árboles).")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fracción del conjunto de desarrollo (solo para pruebas rápidas).")
    parser.add_argument("--recompute-split", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("matplotlib", "PIL", "shap", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    config.set_global_seed()
    globals()[f"cmd_{args.step}"](args)


if __name__ == "__main__":
    main()
