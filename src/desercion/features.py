"""Variables del modelo, variables construidas y preprocesamiento (siempre dentro de un Pipeline)."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from desercion import config

# Variables candidatas: disponibles al terminar el primer semestre y utilizables en AD19–AD20.
CANDIDATES: dict[str, list[str]] = {
    "num": [
        "age", "scholarship.perc", "loan.perc", "PNA", "admission.test", "admission.test.online",
        "admission.rubric", "general.math.eval", "english.evaluation", "FTE",
        "average.first.period", "failed.subject.first.period", "dropped.subject.first.period",
    ],
    "cat": [
        "gender", "foreign", "max.degree.parents", "father.education.summary",
        "mother.education.summary", "parents.exatec", "first.generation", "zone.type",
        "school.cost", "scholarship.type", "tec.no.tec", "school", "program", "region",
    ],
}

# Variables construidas a partir de otras del mismo estudiante (transformación fila a fila, sin
# parámetros aprendidos de los datos, por lo que no pueden filtrar información entre pliegues).
ENGINEERED: dict[str, str] = {
    "brecha.promedio": "Promedio del 1.er semestre menos promedio de bachillerato.",
    "materias.problema": "Materias reprobadas más materias retiradas en el 1.er semestre.",
}

# Se conserva solo la brecha de promedios: mejora el PR-AUC de LightGBM en los cinco pliegues,
# y añadir también las materias con problema no suma (reports/tables/variables_construidas.csv).
KEPT_ENGINEERED: list[str] = ["brecha.promedio"]

# Variables descartadas en la selección: su importancia por permutación media en los cinco
# pliegues es <= 0 (reports/tables/seleccion_importancias.csv) y quitarlas no cambia el PR-AUC
# en particiones nuevas (seleccion_verificacion.csv).
DROPPED_BY_SELECTION: list[str] = [
    "admission.test.online", "first.generation", "foreign", "tec.no.tec", "zone.type",
]

# Variables excluidas antes de modelar y motivo (se documenta en el informe y en reports/tables).
EXCLUDED = {
    "retention": "Es el objetivo (se recodifica como desercion = 1 - retention).",
    "dropout.semester": "Derivada del objetivo: solo es > 0 si el estudiante desertó (fuga directa).",
    "student.id": "Identificador; se usa solo para agrupar la partición.",
    "generation": "Identifica la cohorte; una cohorte nueva no existiría en entrenamiento.",
    "level": "Constante tras filtrar pregrado.",
    "educational.model": "Constante en AD19–AD20 (todos los estudiantes cursan el mismo modelo).",
    "socioeconomic.level": "No utilizable: falta en el 55 % de AD19–AD20 porque no se registró para "
                           "todos los estudiantes, y la tasa de deserción es igual con y sin dato.",
    "social.lag": "No utilizable: falta en el 44 % de AD19–AD20 (mismo motivo que el nivel "
                  "socioeconómico) y no se asocia con la deserción.",
    "total.scholarship.loan": "Redundante: es la suma de % de beca y % de crédito (VIF de 10^7).",
    "online.test": "Redundante: indica si el examen de admisión se presentó en línea, lo que ya "
                   "codifica la variable admission.test.online.",
    "id.school.origin": "6.534 niveles; su información se resume en school.cost y tec.no.tec.",
    "father.education.complete": "Redundante con father.education.summary.",
    "mother.education.complete": "Redundante con mother.education.summary.",
    "father.exatec": "Redundante con parents.exatec.",
    "mother.exatec": "Redundante con parents.exatec.",
    "physical.education": "Actividad co-curricular medida durante el año: refleja la permanencia "
                          "en lugar de anticiparla.",
    "cultural.diffusion": "Ídem actividades co-curriculares.",
    "student.society": "Ídem actividades co-curriculares.",
    "total.life.activities": "Ídem actividades co-curriculares.",
    "athletic.sports": "Ídem actividades co-curriculares.",
    "art.culture": "Ídem actividades co-curriculares.",
    "student.society.leadership": "Ídem actividades co-curriculares.",
    "life.work.mentoring": "Ídem actividades co-curriculares.",
    "wellness.activities": "Ídem actividades co-curriculares.",
}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables construidas (operaciones fila a fila)."""
    df = df.copy()
    df["brecha.promedio"] = df["average.first.period"] - df["PNA"]
    df["materias.problema"] = df["failed.subject.first.period"] + df["dropped.subject.first.period"]
    return df


def feature_lists(engineered: list[str] | None = None, selected: bool = True,
                  drop: list[str] | None = None) -> tuple[list[str], list[str]]:
    """(numéricas, categóricas) del modelo.

    ``engineered`` son las variables construidas que se añaden (por defecto, las conservadas);
    ``selected`` quita las descartadas en la selección; ``drop`` quita variables adicionales.
    """
    num = CANDIDATES["num"] + (KEPT_ENGINEERED if engineered is None else engineered)
    cat = list(CANDIDATES["cat"])
    out = set(drop or []) | (set(DROPPED_BY_SELECTION) if selected else set())
    return [c for c in num if c not in out], [c for c in cat if c not in out]


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Imputación + escalamiento (numéricas) y one-hot (categóricas).

    Todo se ajusta con ``fit`` sobre la partición de entrenamiento de cada pliegue: medianas,
    medias/desviaciones y categorías nunca ven los datos de validación.
    """
    min_freq = config.load_config()["onehot_min_frequency"]
    numeric = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="constant", fill_value="No information")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=min_freq, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, num), ("cat", categorical, cat)],
        verbose_feature_names_out=False,
    )


def make_native_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Preprocesamiento mínimo para CatBoost con categóricas nativas (sin one-hot ni escalamiento)."""
    ct = ColumnTransformer(
        [
            ("num", "passthrough", num),
            ("cat", SimpleImputer(strategy="constant", fill_value="No information"), cat),
        ],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


# Nombre legible de cada variable para figuras y tablas del informe.
VAR_LABELS = {
    "age": "Edad",
    "gender": "Género",
    "foreign": "Procedencia",
    "scholarship.perc": "% de beca",
    "loan.perc": "% de crédito",
    "total.scholarship.loan": "% total de apoyo (beca + crédito)",
    "max.degree.parents": "Máximo grado de los padres",
    "father.education.summary": "Educación del padre",
    "mother.education.summary": "Educación de la madre",
    "parents.exatec": "Padres exalumnos del Tec",
    "first.generation": "Primera generación universitaria",
    "zone.type": "Tipo de zona de residencia",
    "socioeconomic.level": "Nivel socioeconómico",
    "social.lag": "Rezago social",
    "school.cost": "Costo del colegio de origen",
    "scholarship.type": "Tipo de beca",
    "PNA": "Promedio de bachillerato",
    "admission.test": "Examen de admisión (presencial)",
    "admission.test.online": "Examen de admisión (en línea)",
    "admission.rubric": "Rúbrica de admisión",
    "general.math.eval": "Nota de matemáticas (admisión)",
    "english.evaluation": "Nivel de inglés",
    "online.test": "Examen en línea",
    "tec.no.tec": "Colegio del Tec",
    "FTE": "Carga académica",
    "educational.model": "Modelo educativo",
    "school": "Escuela",
    "program": "Programa",
    "region": "Región del campus",
    "average.first.period": "Promedio del 1.er semestre",
    "failed.subject.first.period": "Materias reprobadas (1.er sem.)",
    "dropped.subject.first.period": "Materias retiradas (1.er sem.)",
    "brecha.promedio": "Brecha promedio 1.er sem. − bachillerato",
    "materias.problema": "Materias reprobadas + retiradas",
}
