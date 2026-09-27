"""Selección de variables por escenario y preprocesamiento (siempre dentro de un Pipeline)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from desercion import config

# Grupos temáticos (se usan también en el análisis de ablación de la pregunta de investigación).
GROUPS: dict[str, dict[str, list[str]]] = {
    "demograficas": {
        "num": ["age"],
        "cat": ["gender", "foreign"],
    },
    "socioeconomicas": {
        "num": ["scholarship.perc", "loan.perc", "total.scholarship.loan"],
        "cat": [
            "max.degree.parents", "father.education.summary", "mother.education.summary",
            "parents.exatec", "first.generation", "zone.type", "socioeconomic.level",
            "social.lag", "school.cost", "scholarship.type",
        ],
    },
    "academicas_ingreso": {
        "num": ["PNA", "admission.test", "admission.rubric", "general.math.eval",
                "english.evaluation", "online.test"],
        "cat": ["tec.no.tec"],
    },
    "institucionales": {
        "num": ["FTE", "educational.model"],
        "cat": ["school", "program", "region"],
    },
    "primer_periodo": {
        "num": ["average.first.period", "failed.subject.first.period",
                "dropped.subject.first.period"],
        "cat": [],
    },
}

SCENARIO_GROUPS = {
    "A": ["demograficas", "socioeconomicas", "academicas_ingreso", "institucionales"],
    "B": ["demograficas", "socioeconomicas", "academicas_ingreso", "institucionales", "primer_periodo"],
}

# Variables excluidas y motivo (se documenta en el informe y en reports/tables).
EXCLUDED = {
    "retention": "Es el objetivo (se recodifica como desercion = 1 - retention).",
    "dropout.semester": "Derivada del objetivo: solo es > 0 si el estudiante desertó (fuga directa).",
    "student.id": "Identificador; se usa solo para agrupar la partición.",
    "generation": "Identifica la cohorte; una cohorte nueva no existiría en entrenamiento.",
    "level": "Constante tras filtrar pregrado.",
    "id.school.origin": "6.534 niveles; su información se resume en school.cost y tec.no.tec.",
    "father.education.complete": "Redundante con father.education.summary.",
    "mother.education.complete": "Redundante con mother.education.summary.",
    "father.exatec": "Redundante con parents.exatec.",
    "mother.exatec": "Redundante con parents.exatec.",
    "physical.education": "Actividad co-curricular del periodo: ambigüedad temporal y 'No information' "
                          "concentra desertores (marcador de fuga).",
    "cultural.diffusion": "Ídem actividades co-curriculares.",
    "student.society": "Ídem actividades co-curriculares.",
    "total.life.activities": "Ídem actividades co-curriculares.",
    "athletic.sports": "Ídem actividades co-curriculares.",
    "art.culture": "Ídem actividades co-curriculares.",
    "student.society.leadership": "Ídem actividades co-curriculares.",
    "life.work.mentoring": "Ídem actividades co-curriculares.",
    "wellness.activities": "Ídem actividades co-curriculares.",
}


def feature_lists(scenario: str, groups: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Devuelve (numéricas, categóricas) para un escenario o un subconjunto de grupos."""
    groups = SCENARIO_GROUPS[scenario] if groups is None else groups
    num = [c for g in groups for c in GROUPS[g]["num"]]
    cat = [c for g in groups for c in GROUPS[g]["cat"]]
    return num, cat


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
