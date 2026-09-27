# Genera notebooks/01_exploracion_datos.ipynb (se mantiene como .py para revisar diffs con facilidad).
import nbformat as nbf

cells = [
    nbf.v4.new_markdown_cell(
        "# 01 · Exploración y consolidación de los datos\n\n"
        "Notebook de apoyo a la sección 1 del informe de la Segunda entrega. Toda la lógica vive en "
        "`src/desercion`; aquí solo se inspecciona. Requiere haber ejecutado `python -m desercion.cli prepare`."
    ),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from desercion import data\n"
        "from desercion.features import feature_lists, EXCLUDED\n"
        "pd.set_option('display.width', 160)\n"
        "df = data.load_clean()\n"
        "split = data.get_split(df)\n"
        "df.shape"
    ),
    nbf.v4.new_markdown_cell("## Objetivo: deserción en el primer año por cohorte y momento"),
    nbf.v4.new_code_cell(
        "pd.concat([df.groupby('generation')[data.TARGET].agg(['size', 'mean']),\n"
        "           pd.crosstab(df.generation, df['dropout.semester'])], axis=1)"
    ),
    nbf.v4.new_markdown_cell(
        "## Evidencia de fuga en las actividades co-curriculares\n"
        "El valor *No information* concentra desertores: el registro falta *porque* el estudiante se fue."
    ),
    nbf.v4.new_code_cell(
        "pd.crosstab(df['total.life.activities'], df[data.TARGET], normalize='index').round(3)"
    ),
    nbf.v4.new_markdown_cell(
        "## Desempeño del primer periodo según el momento de la deserción (AD19–AD20)\n"
        "Quienes desertan *durante* el primer semestre (código 1) tienen promedios muy bajos porque el "
        "abandono ocurre dentro del periodo; por eso se excluyen del escenario B."
    ),
    nbf.v4.new_code_cell(
        "b = df[df.generation.isin(['AD19', 'AD20'])]\n"
        "b.groupby('dropout.semester')[['average.first.period', 'failed.subject.first.period',\n"
        "                               'dropped.subject.first.period']].mean().round(2)"
    ),
    nbf.v4.new_markdown_cell("## Tamaños por escenario y partición"),
    nbf.v4.new_code_cell("pd.read_csv('../reports/tables/datos_resumen.csv')"),
    nbf.v4.new_markdown_cell("## Variables excluidas y motivo"),
    nbf.v4.new_code_cell("pd.Series(EXCLUDED, name='motivo').to_frame()"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3",
                                                                "display_name": "Python 3"}})
nbf.write(nb, "notebooks/01_exploracion_datos.ipynb")
