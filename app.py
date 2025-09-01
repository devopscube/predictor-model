import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "/app/model/model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI()

class PredictIn(BaseModel):
    instances: list

@app.get("/healthz")
def healthz():
    return {"ok": True, "model_path": MODEL_PATH}

@app.post("/v1/models/model:predict")
def predict(p: PredictIn):
    X = p.instances
    try:
        X_arr = np.array(X, dtype=float)
        preds = model.predict(X_arr).tolist()
    except Exception:
        preds = model.predict(X).tolist()
    return {"predictions": preds}
