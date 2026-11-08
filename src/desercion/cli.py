"""Punto de entrada del flujo reproducible.

Uso (desde la raíz del repositorio, con el entorno activado):

    python -m desercion.cli prepare       # limpieza, partición sellada, calidad de datos
    python -m desercion.cli eda           # análisis exploratorio del conjunto de desarrollo
    python -m desercion.cli select        # variables construidas y selección de variables
    python -m desercion.cli train         # CV anidada de todos los modelos
    python -m desercion.cli compare       # pruebas estadísticas entre modelos
    python -m desercion.cli experiments   # réplicas del artículo: SMOTE/ADASYN, fuga, Isolation Forest, VIF
    python -m desercion.cli interpret     # SHAP por pliegue y análisis de errores
    python -m desercion.cli figures       # figuras del informe
    python -m desercion.cli report        # tablas LaTeX del informe (reports/tables)
    python -m desercion.cli final         # modelo final: evaluación única en prueba y artefactos
    python -m desercion.cli all           # todo lo anterior, en orden
"""

from __future__ import annotations

import argparse
import logging
import warnings

import pandas as pd

from desercion import config, data, eda, evaluate, experiments, final, interpret, plots, report
from desercion.features import (
    CANDIDATES,
    DROPPED_BY_SELECTION,
    EXCLUDED,
    KEPT_ENGINEERED,
    add_engineered,
    feature_lists,
)
from desercion.models import get_specs, params_from_text

log = logging.getLogger("desercion")
TREE_MODELS = ["LightGBM", "CatBoost", "XGBoost", "Gradient Boosting", "Random Forest"]
BASELINES = ("Mayoría", "Árbol sin poda")


def _tables():
    return config.path("tables_dir")


def _figures():
    return config.path("figures_dir")


def _dev(sample: float | None = None) -> pd.DataFrame:
    """Conjunto de desarrollo de la población de estudio, con las variables construidas."""
    df = data.load_clean()
    dev = add_engineered(data.study_frame(df, data.get_split(df), "dev"))
    if sample:
        dev = dev.groupby(data.TARGET, group_keys=False).sample(frac=sample,
                                                               random_state=config.seed())
    return dev


def _folds() -> pd.DataFrame:
    return pd.read_csv(_tables() / "cv_pliegues.csv")


def _ranking(folds: pd.DataFrame) -> pd.Series:
    return folds.groupby("modelo")["pr_auc"].mean().sort_values(ascending=False)


def _chosen(folds: pd.DataFrame, name: str) -> dict:
    """Hiperparámetros elegidos con más frecuencia en los pliegues externos."""
    return params_from_text(folds.loc[folds.modelo == name, "params"].mode().iloc[0])


def _interpret_model(folds: pd.DataFrame, forced: str | None = None) -> str:
    """Mejor modelo de árboles (SHAP exacto con TreeExplainer)."""
    return forced or next(m for m in _ranking(folds).index if m in TREE_MODELS)


# ----------------------------------------------------------------------- pasos
def cmd_prepare(args) -> None:
    df = data.load_clean()
    split = data.get_split(df, recompute=args.recompute_split)
    rows = []
    for part in ("dev", "test"):
        f = data.study_frame(df, split, part)
        rows.append({"parte": part, "registros": len(f), "estudiantes": f[data.GROUP].nunique(),
                     "desertores": int(f[data.TARGET].sum()), "tasa_desercion": f[data.TARGET].mean()})
    resumen = pd.DataFrame(rows)
    resumen.to_csv(_tables() / "datos_resumen.csv", index=False)
    log.info("\n%s", resumen.to_string(index=False))
    both = df.loc[split == "dev", data.GROUP].isin(df.loc[split == "test", data.GROUP]).sum()
    log.info("Estudiantes presentes en desarrollo y prueba a la vez: %d", both)

    dev = data.study_frame(df, split, "dev")
    q = data.quality_report(dev)
    q.to_csv(_tables() / "calidad_datos.csv", index=False)
    pd.Series(EXCLUDED, name="motivo").rename_axis("variable").to_csv(
        _tables() / "variables_excluidas.csv")
    plots.data_quality(q, CANDIDATES["num"] + CANDIDATES["cat"] + ["socioeconomic.level",
                                                                    "social.lag"],
                       _figures() / "fig01_calidad_datos.png")


EDA_NUMERIC = ["average.first.period", "PNA", "general.math.eval", "admission.test",
               "scholarship.perc", "age", "FTE", "brecha.promedio"]
EDA_CATEGORICAL = ["scholarship.type", "program", "region", "school.cost", "school",
                   "father.education.summary"]


def cmd_eda(args) -> None:
    """EDA del conjunto de desarrollo (el de prueba sellado no se mira)."""
    df = data.load_clean()
    split = data.get_split(df)
    dev = add_engineered(data.study_frame(df, split, "dev"))
    t, f = _tables() / "eda", _figures() / "eda"
    t.mkdir(exist_ok=True)
    f.mkdir(exist_ok=True)
    for old in list(t.glob("*.csv")) + list(f.glob("*.png")):
        old.unlink()
    overall = dev[data.TARGET].mean()
    num, cat = CANDIDATES["num"] + KEPT_ENGINEERED, CANDIDATES["cat"]

    # Contexto: todo el pregrado de desarrollo, para justificar la elección de cohortes.
    undergrad = df[split == "dev"]
    cohort = eda.target_by_cohort(undergrad)
    cohort.to_csv(t / "objetivo_por_cohorte.csv")
    plots.eda_dropout_moment(cohort, f / "eda01_desercion_momento.png")
    context = [c for c in num + cat if c != "admission.test.online"] + ["socioeconomic.level",
                                                                       "social.lag"]
    miss = eda.missing_by_cohort(undergrad, [c for c in context if c in undergrad])
    miss.to_csv(t / "faltantes_por_cohorte.csv")
    plots.eda_missing_heatmap(miss, f / "eda02_faltantes_cohorte.png")

    # Población de estudio (AD19–AD20).
    mvt = eda.missing_vs_target(dev, num + cat + ["socioeconomic.level", "social.lag"])
    mvt.to_csv(t / "faltantes_vs_desercion.csv", index=False)
    plots.eda_missing_vs_target(mvt, overall, f / "eda03_faltantes_vs_desercion.png")

    profiles = {c: eda.rate_by_quantile(dev, c) for c in EDA_NUMERIC}
    pd.concat(profiles, names=["variable", "intervalo"]).to_csv(t / "perfil_numericas.csv")
    plots.eda_numeric_profiles(profiles, overall, f / "eda04_perfil_numericas.png")

    cats = {c: eda.rate_by_category(dev, c) for c in EDA_CATEGORICAL}
    pd.concat(cats, names=["variable", "categoria"]).to_csv(t / "perfil_categoricas.csv")
    plots.eda_categorical_profiles(cats, overall, f / "eda05_perfil_categoricas.png")

    assoc = eda.univariate_association(dev, num, cat)
    assoc.to_csv(t / "asociacion_univariada.csv", index=False)
    plots.eda_association({"Población de estudio (AD19–AD20)": assoc},
                          ["average.first.period", "failed.subject.first.period",
                           "dropped.subject.first.period", "brecha.promedio"],
                          f / "eda06_asociacion_univariada.png")

    eda.numeric_summary(dev, num).to_csv(t / "resumen_numericas.csv", index=False)
    corr = eda.numeric_correlation(dev, num + ["total.scholarship.loan"])
    corr.to_csv(t / "correlacion_spearman.csv")
    plots.eda_correlation(corr, f / "eda07_correlacion.png")
    log.info("EDA listo: %d tablas en %s", len(list(t.glob("*.csv"))), t)


def cmd_select(args) -> None:
    """Variables construidas y selección de variables (insumo de features.py)."""
    t, dev = _tables(), _dev(args.sample)
    experiments.feature_engineering_check(dev).to_csv(t / "variables_construidas.csv", index=False)
    imp, scores = experiments.feature_selection(dev)
    imp.to_csv(t / "seleccion_importancias.csv", index=False)
    scores.to_csv(t / "seleccion_desempeno.csv", index=False)
    drop = imp.groupby("variable")["importancia"].mean().loc[lambda s: s <= 0].index.tolist()
    log.info("Variables con importancia media <= 0: %s (en features.py: %s)", drop,
             DROPPED_BY_SELECTION)
    experiments.selection_check(dev, drop).to_csv(t / "seleccion_verificacion.csv", index=False)


def cmd_train(args) -> None:
    specs = get_specs()
    names = args.models or list(specs)
    dev = _dev(args.sample)
    num, cat = feature_lists()
    X, y, g = dev[num + cat], dev[data.TARGET], dev[data.GROUP]
    log.info("%d registros, %d variables, %.2f%% desertores", len(dev), len(num) + len(cat),
             100 * y.mean())
    f_path = _tables() / "cv_pliegues.csv"
    o_path = config.path("processed_dir") / "oof.parquet"
    folds = pd.read_csv(f_path) if f_path.exists() and args.models else pd.DataFrame()
    oofs = pd.read_parquet(o_path) if o_path.exists() and args.models else pd.DataFrame()
    for name in names:
        fr, oo = evaluate.nested_cv(specs[name], X, y, g, num, cat)
        if not folds.empty:
            folds, oofs = folds[folds.modelo != name], oofs[oofs.modelo != name]
        folds = pd.concat([folds, fr], ignore_index=True)
        oofs = pd.concat([oofs, oo], ignore_index=True)
        folds.to_csv(f_path, index=False)
        oofs.to_parquet(o_path)
    summary = evaluate.summarize(folds)
    summary.to_csv(_tables() / "cv_resumen.csv")
    log.info("\n%s", summary[["pr_auc_mean", "pr_auc_std", "f2_mean", "recall_mean",
                              "precision_mean"]].round(3))


def cmd_compare(args) -> None:
    """Mejor modelo contra cada uno de los demás (t corregida de Nadeau y Bengio)."""
    folds = _folds()
    k = config.load_config()["cv"]["outer_folds"]
    n = len(_dev())
    n_test = n // k
    best = _ranking(folds).index[0]
    rows = [{**evaluate.compare_models(folds, best, other, metric, n - n_test, n_test)}
            for other in _ranking(folds).index[1:] for metric in ("pr_auc", "f2")]
    out = pd.DataFrame(rows)
    out.to_csv(_tables() / "pruebas_estadisticas.csv", index=False)
    log.info("\n%s", out.round(4).to_string(index=False))


EXPERIMENTS = ["balanceo_arbol", "fuga", "isolation_forest", "vif", "curva_validacion", "costo",
               "sensibilidad"]
FINAL_MODEL = "LightGBM"

# Curvas de validación: hiperparámetro de complejidad de un modelo de boosting y del SVM.
CURVES = {"LightGBM": ("clf__num_leaves", [2, 4, 8, 16, 32, 64, 128]),
          "SVM (RBF-Nystroem)": ("clf__svm__C", [1e-4, 1e-3, 1e-2, 1e-1, 1.0])}


def cmd_experiments(args) -> None:
    t, dev = _tables(), _dev(args.sample)
    todo = set(args.experiments or EXPERIMENTS)
    if "balanceo_arbol" in todo:
        experiments.resampling_comparison(dev).to_csv(t / "balanceo_arbol.csv", index=False)
    if "fuga" in todo:
        experiments.leakage_demo(dev).to_csv(t / "fuga_smote.csv", index=False)
    if "isolation_forest" in todo:
        experiments.isolation_forest(dev).to_csv(t / "isolation_forest.csv", index=False)
    if "vif" in todo:
        experiments.vif(dev).to_csv(t / "vif.csv", index=False)
        experiments.vif(dev, extra=["total.scholarship.loan"]).to_csv(
            t / "vif_con_apoyo_total.csv", index=False)
    if {"curva_validacion", "costo"} & todo:
        folds, specs = _folds(), get_specs()
    if "curva_validacion" in todo:
        curves = [experiments.validation_curve(dev, specs[m], _chosen(folds, m), p, v)
                  for m, (p, v) in CURVES.items()]
        pd.concat(curves).to_csv(t / "curva_validacion.csv", index=False)
    if "costo" in todo:
        experiments.computational_cost(dev, folds, specs).to_csv(t / "costo_computacional.csv",
                                                                 index=False)
    if "sensibilidad" in todo:
        spec = get_specs()[FINAL_MODEL]
        s_folds, s_oof = experiments.missing_flag_sensitivity(dev, spec)
        s_folds.to_csv(t / "sensibilidad_pliegues.csv", index=False)
        s_oof.to_parquet(config.path("processed_dir") / "oof_sensibilidad.parquet")
        temporal = experiments.temporal_validation(dev, spec)
        reference = experiments.cohort_reference(s_oof, dev)
        pd.concat([temporal, reference], ignore_index=True).to_csv(
            t / "validacion_temporal.csv", index=False)
        per_fold, curve = experiments.calibration(
            s_oof[s_oof.variante == experiments.FINAL_VARIANTS[0]])
        per_fold.to_csv(t / "calibracion.csv", index=False)
        curve.to_csv(t / "calibracion_curva.csv", index=False)
    log.info("Experimentos listos")


def cmd_interpret(args) -> None:
    t, dev, folds = _tables(), _dev(args.sample), _folds()
    name = _interpret_model(folds, args.model)
    spec, params = get_specs()[name], _chosen(folds, name)
    log.info("Interpretando %s con %s", name, params)
    table, last = interpret.shap_by_fold(dev, spec, params)
    table.to_csv(t / "shap.csv")
    pd.to_pickle(last, config.path("processed_dir") / "shap_ultimo_pliegue.pkl")

    oof_all = pd.read_parquet(config.path("processed_dir") / "oof.parquet")
    evaluate.capture_at_top(oof_all).to_csv(t / "captura_topk.csv", index=False)
    best = _ranking(folds).index[0]
    oof = oof_all[oof_all.modelo == best]
    segs = ["scholarship.type", "school", "region", "program"]
    interpret.error_analysis(oof, dev, segs).to_csv(t / "errores.csv", index=False)
    cm = pd.crosstab(oof.y.map({0: "Retenido", 1: "Deserción"}),
                     oof.pred.map({0: "Pred. retenido", 1: "Pred. deserción"}))
    cm.to_csv(t / "confusion.csv")
    log.info("Matriz de confusión (%s):\n%s", best, cm)


def cmd_figures(args) -> None:
    t, f = _tables(), _figures()
    prev = pd.read_csv(t / "datos_resumen.csv").query("parte == 'dev'")["tasa_desercion"].iloc[0]
    if (t / "seleccion_importancias.csv").exists():
        plots.selection_importance(pd.read_csv(t / "seleccion_importancias.csv"),
                                   DROPPED_BY_SELECTION, f / "fig02_seleccion_variables.png")
    if (t / "cv_pliegues.csv").exists():
        folds = _folds()
        plots.model_comparison(folds, prev, f / "fig03_comparacion_modelos.png")
        oof = pd.read_parquet(config.path("processed_dir") / "oof.parquet")
        top = [m for m in _ranking(folds).index if m not in BASELINES][:3] + ["Árbol sin poda"]
        plots.pr_curves(oof, top, prev, f / "fig04_curvas_pr.png")
    if (t / "fuga_smote.csv").exists():
        plots.leakage(pd.read_csv(t / "fuga_smote.csv"), f / "fig05_fuga_smote.png")
    if (t / "curva_validacion.csv").exists():
        plots.validation_curve(pd.read_csv(t / "curva_validacion.csv"),
                               f / "fig06_curva_validacion.png")
    if (t / "costo_computacional.csv").exists() and (t / "cv_resumen.csv").exists():
        plots.cost_vs_performance(pd.read_csv(t / "cv_resumen.csv", index_col=0),
                                  pd.read_csv(t / "costo_computacional.csv"),
                                  f / "fig09_costo_desempeno.png")
    if (t / "shap.csv").exists():
        name = _interpret_model(_folds(), args.model)
        plots.shap_importance(pd.read_csv(t / "shap.csv", index_col=0),
                              f / "fig07_shap_importancia.png", name)
        last = config.path("processed_dir") / "shap_ultimo_pliegue.pkl"
        if last.exists():
            plots.shap_beeswarm(pd.read_pickle(last), f / "fig08_shap_beeswarm.png")


def cmd_report(args) -> None:
    """Tablas LaTeX del informe, generadas desde los resultados (el .tex las lee de reports/)."""
    t, folds = _tables(), _folds()
    summary = pd.read_csv(t / "cv_resumen.csv", index_col=0)
    grids = {name: spec.grid for name, spec in get_specs().items()}
    report.results_table(summary, t / "tabla_resultados.tex")
    report.hyperparams_table(folds, grids, t / "tabla_hiperparametros.tex")
    report.leakage_table(pd.read_csv(t / "fuga_smote.csv"), t / "tabla_fuga.tex")
    report.cost_table(pd.read_csv(t / "costo_computacional.csv"), summary, t / "tabla_costo.tex")
    log.info("Tablas LaTeX listas en %s", t)


def cmd_final(args) -> None:
    """Ajusta el modelo final con todo el desarrollo, lo evalúa una vez en prueba y lo exporta."""
    t = _tables()
    done = t / "test_final.csv"
    if done.exists() and not args.force:
        log.warning("El conjunto de prueba ya se evaluó (%s). Para reproducir la evaluación "
                    "con el mismo modelo congelado, use --force.", done)
        return
    df = data.load_clean()
    split = data.get_split(df)
    dev = add_engineered(data.study_frame(df, split, "dev"))
    test = add_engineered(data.study_frame(df, split, "test"))
    fit = final.fit_final(dev)
    metrics, by_cohort = final.evaluate_test(fit, test)
    metrics.to_csv(done, index=False)
    by_cohort.to_csv(t / "test_por_cohorte.csv", index=False)
    X_te = test[fit["num"] + fit["cat"]]
    groups = final.shap_groups(fit["model"], fit["num"], fit["cat"])
    sv = final.shap_by_variable(fit["model"], X_te, groups)
    sv.abs().mean().sort_values(ascending=False).rename("shap_abs_medio").to_csv(
        t / "shap_final_test.csv")
    pd.DataFrame({"row_id": test.index, "generation": test["generation"].to_numpy(),
                  "y": test[data.TARGET].to_numpy(),
                  "score": fit["model"].predict_proba(X_te)[:, 1]}).to_parquet(
        config.path("processed_dir") / "test_pred.parquet")
    final.export_artifacts(fit, dev, metrics, config.path("deploy_artifacts"))
    log.info("Prueba: %s", metrics.set_index("metrica")["valor"].round(3).to_dict())


STEPS = ["prepare", "eda", "select", "train", "compare", "experiments", "interpret", "figures",
         "report", "final"]


def cmd_all(args) -> None:
    for step in STEPS:
        globals()[f"cmd_{step}"](args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="desercion", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS + ["all"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Re-entrena solo estos modelos y conserva los demás resultados.")
    parser.add_argument("--experiments", nargs="+", default=None, choices=EXPERIMENTS,
                        help="Subconjunto de experimentos (por defecto, todos).")
    parser.add_argument("--model", default=None, help="Modelo a interpretar (por defecto, el "
                                                      "mejor de árboles).")
    parser.add_argument("--sample", type=float, default=None,
                        help="Fracción del conjunto de desarrollo (solo para pruebas rápidas).")
    parser.add_argument("--recompute-split", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Repite la evaluación en prueba aunque ya exista (mismo modelo).")
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
