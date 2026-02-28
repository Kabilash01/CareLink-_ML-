import numpy as np
import pandas as pd
import random
from pathlib import Path


class StructuredTriageGenerator:

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        # Weighted by clinical severity (higher weight = more dangerous)
        self.high_symptoms = [
            ("chest pain",          4.0),
            ("shortness of breath", 3.5),
            ("unconsciousness",     5.0),
            ("severe bleeding",     4.5),
            ("seizure",             4.5),
            ("paralysis",           4.5),
            ("stroke symptoms",     5.0),
            ("severe allergic reaction", 4.0),
            ("heart palpitations",  3.5),
            ("coughing blood",      4.0),
        ]

        self.medium_symptoms = [
            ("high fever",          2.5),
            ("vomiting",            2.0),
            ("abdominal pain",      2.5),
            ("migraine",            2.0),
            ("dizziness",           2.0),
            ("joint swelling",      2.0),
            ("difficulty swallowing", 2.5),
            ("painful urination",   1.5),
            ("nausea",              1.5),
            ("rash",                1.5),
        ]

        self.low_symptoms = [
            ("cough",               0.5),
            ("runny nose",          0.3),
            ("sore throat",         0.5),
            ("fatigue",             0.5),
            ("mild headache",       0.5),
            ("sneezing",            0.3),
            ("mild back pain",      0.5),
            ("loss of appetite",    0.5),
            ("insomnia",            0.4),
            ("dry skin",            0.2),
        ]

        # (name, risk_weight)
        self.chronic_conditions = [
            ("diabetes",       1.5),
            ("heart disease",  2.5),
            ("asthma",         1.5),
            ("hypertension",   2.0),
            ("COPD",           2.0),
            ("kidney disease", 2.0),
            ("cancer",         2.5),
            ("stroke history", 2.5),
            ("obesity",        1.0),
            ("anemia",         1.0),
        ]

    def generate_sample(self):

        age            = np.random.randint(1, 95)
        duration_days  = int(np.random.exponential(scale=5)) + 1
        duration_days  = min(duration_days, 365)

        # High: rare spikes (right-skewed)
        high_count   = np.random.choice([0, 1, 2, 3], p=[0.55, 0.28, 0.12, 0.05])
        medium_count = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10])
        low_count    = np.random.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10])

        picked_high   = random.sample(self.high_symptoms,   min(high_count,   len(self.high_symptoms)))
        picked_medium = random.sample(self.medium_symptoms, min(medium_count, len(self.medium_symptoms)))
        picked_low    = random.sample(self.low_symptoms,    min(low_count,    len(self.low_symptoms)))

        high_weight   = sum(w for _, w in picked_high)
        medium_weight = sum(w for _, w in picked_medium)
        low_weight    = sum(w for _, w in picked_low)

        # Chronic conditions
        chronic_count = np.random.choice([0, 1, 2, 3], p=[0.50, 0.30, 0.15, 0.05])
        picked_chronic = random.sample(self.chronic_conditions, min(chronic_count, len(self.chronic_conditions)))
        chronic_weight = sum(w for _, w in picked_chronic)

        # Age-based risk factor (elderly & very young at higher risk)
        if age >= 70:
            age_factor = 2.5
        elif age >= 60:
            age_factor = 1.8
        elif age >= 50:
            age_factor = 1.2
        elif age <= 5:
            age_factor = 1.5
        else:
            age_factor = 0.0

        # Duration modifier
        if duration_days >= 14:
            duration_factor = 1.0
        elif duration_days >= 7:
            duration_factor = 0.5
        else:
            duration_factor = 0.0

        risk_score = (
            high_weight   * 1.0 +
            medium_weight * 0.5 +
            low_weight    * 0.2 +
            chronic_weight * 0.6 +
            age_factor +
            duration_factor +
            np.random.normal(0, 0.3)   # small Gaussian noise
        )

        if risk_score >= 6.0:
            risk = "HIGH"
        elif risk_score >= 2.5:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "age":                  age,
            "duration_days":        duration_days,
            "num_high_symptoms":    high_count,
            "num_medium_symptoms":  medium_count,
            "num_low_symptoms":     low_count,
            "chronic_count":        chronic_count,
            "high_weight":          round(high_weight,   2),
            "medium_weight":        round(medium_weight, 2),
            "chronic_weight":       round(chronic_weight,2),
            "age_factor":           round(age_factor,    2),
            "duration_factor":      round(duration_factor, 2),
            "risk_score_internal":  round(risk_score,    2),
            "is_elderly":           int(age >= 65),
            "is_child":             int(age <= 12),
            "has_critical_symptom": int(high_count >= 1),
            "risk_level":           risk,
        }

    def generate_dataset(self, n=50000):
        data = [self.generate_sample() for _ in range(n)]
        return pd.DataFrame(data)


if __name__ == "__main__":
    gen = StructuredTriageGenerator()
    df  = gen.generate_dataset(50000)

    Path("data").mkdir(exist_ok=True)
    out = "data/structured_training_dataset.csv"
    df.to_csv(out, index=False)

    print(f"Dataset generated: {len(df)} samples → {out}")
    print(df["risk_level"].value_counts())
    print(df.describe())
