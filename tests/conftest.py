"""Datos sintéticos con el mismo esquema del dataset del TEC (las pruebas no usan datos reales)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from desercion.features import GROUPS


@pytest.fixture(scope="session")
def raw_like() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    cats = {
        "gender": ["Male", "Female"],
        "foreign": ["Local", "Yes: National", "Yes: Foreigner"],
        "max.degree.parents": ["No information", "No degree", "Undergraduate degree", "Master degree"],
        "father.education.summary": ["No information", "Undergraduate degree"],
        "mother.education.summary": ["No information", "Undergraduate degree"],
        "parents.exatec": ["Yes", "No", "No information"],
        "first.generation": ["Yes", "No", "No information"],
        "zone.type": ["Urban", "Rural", "No information"],
        "socioeconomic.level": ["Level 6", "Level 7", "No information"],
        "social.lag": ["Low", "Medium", "No information"],
        "school.cost": ["High cost", "Public", "Not defined"],
        "scholarship.type": ["No scholarship", "Academic talent", "Traditional"],
        "tec.no.tec": ["TEC", "NO TEC"],
        "school": ["EIC", "EN", "ECSG"],
        "program": ["IIS", "LAE", "ARQ", "MC"],
        "region": ["RM", "RO", "RCM"],
    }
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in cats.items()})
    df["student.id"] = np.arange(n) // 2 * 1 + 1  # algunos estudiantes repetidos
    df["generation"] = rng.choice(["AD18", "AD19", "AD20"], n)
    df["level"] = "Undergraduate"
    df["age"] = rng.integers(17, 22, n)
    df["PNA"] = rng.normal(88, 6, n).round(1)
    df.loc[:5, "PNA"] = 0
    df["admission.test"] = rng.integers(900, 1600, n).astype(str).astype(object)
    df.loc[df.index % 7 == 0, "admission.test"] = "Does not apply "
    df["admission.rubric"] = np.where(rng.random(n) < 0.3, np.nan, rng.integers(0, 50, n))
    df["general.math.eval"] = rng.uniform(0, 100, n).round(1).astype(str).astype(object)
    df.loc[df.index % 11 == 0, "general.math.eval"] = "No information"
    df["english.evaluation"] = rng.integers(0, 8, n)
    df["online.test"] = rng.integers(0, 2, n)
    df["FTE"] = rng.choice([0.8, 1.0, 1.08], n)
    df["educational.model"] = (df["generation"] != "AD18").astype(int)
    df["scholarship.perc"] = rng.choice([0, 0.2, 0.5], n)
    df["loan.perc"] = rng.choice([0, 0.15], n)
    df["total.scholarship.loan"] = df["scholarship.perc"] + df["loan.perc"]
    df["average.first.period"] = rng.uniform(50, 100, n)
    df["failed.subject.first.period"] = rng.integers(0, 3, n)
    df["dropped.subject.first.period"] = rng.integers(0, 3, n)
    risk = (df["average.first.period"] < 70).astype(int) + (rng.random(n) < 0.08)
    df["retention"] = (risk == 0).astype(int)
    df["dropout.semester"] = np.where(df["retention"] == 1, 0, rng.integers(1, 5, n))
    for c in ["id.school.origin", "father.education.complete", "mother.education.complete",
              "father.exatec", "mother.exatec", "physical.education", "total.life.activities"]:
        df[c] = "No information"
    assert all(c in df for g in GROUPS.values() for c in g["num"] + g["cat"])
    df.index.name = "row_id"
    return df
