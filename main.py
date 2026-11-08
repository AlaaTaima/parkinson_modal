from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
MODEL_PATH = BASE / "parkinsons_model.pkl"

# Decision threshold on P(Parkinson's). 0.5 is the default used by model.predict().
THRESHOLD = 0.5

# The 20 columns of X from the notebook, in the same order
# (after dropping name, personNum, status, Jitter:DDP, Shimmer:DDA).
# "hint" is the dataset median (None where it was not shown in the notebook output).
FEATURES = [
    {"name": "MDVP:Fo(Hz)", "hint": 148.79},
    {"name": "MDVP:Fhi(Hz)", "hint": 175.829},
    {"name": "MDVP:Flo(Hz)", "hint": 104.315},
    {"name": "MDVP:Jitter(%)", "hint": 0.00494},
    {"name": "MDVP:Jitter(Abs)", "hint": 0.00003},
    {"name": "MDVP:RAP", "hint": 0.0025},
    {"name": "MDVP:PPQ", "hint": 0.00269},
    {"name": "MDVP:Shimmer", "hint": 0.02297},
    {"name": "MDVP:Shimmer(dB)", "hint": 0.221},
    {"name": "Shimmer:APQ3", "hint": None},
    {"name": "Shimmer:APQ5", "hint": None},
    {"name": "MDVP:APQ", "hint": None},
    {"name": "NHR", "hint": 0.01166},
    {"name": "HNR", "hint": 22.085},
    {"name": "RPDE", "hint": 0.495954},
    {"name": "DFA", "hint": 0.722254},
    {"name": "spread1", "hint": -5.720868},
    {"name": "spread2", "hint": 0.218885},
    {"name": "D2", "hint": 2.361532},
    {"name": "PPE", "hint": 0.194052},
]
NAMES = [f["name"] for f in FEATURES]

if not MODEL_PATH.exists():
    raise RuntimeError(
        "parkinsons_model.pkl not found. Put the file saved by the notebook "
        "next to main.py."
    )

# The saved object is a Pipeline (StandardScaler + LogisticRegression),
# so scaling happens inside the model: do NOT scale the inputs here.
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Parkinson's voice-feature classifier")


class Inp(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return FEATURES


@app.post("/predict")
def predict(d: Inp):
    missing = [n for n in NAMES if n not in d.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    # DataFrame with the training column names and order (avoids sklearn warnings)
    row = pd.DataFrame([[d.features[n] for n in NAMES]], columns=NAMES)
    if not np.isfinite(row.to_numpy()).all():
        raise HTTPException(status_code=422, detail="All values must be finite numbers")

    prob = float(model.predict_proba(row)[0, 1])  # class 1 = Parkinson's
    return {
        "probability": prob,
        "prediction": int(prob >= THRESHOLD),
        "threshold": THRESHOLD,
    }
