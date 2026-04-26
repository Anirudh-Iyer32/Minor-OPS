from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

app = FastAPI()

# ✅ Load model (correct way)
model = joblib.load("models/model.joblib")


class InputData(BaseModel):
    data: List[float]


@app.get("/")
def home():
    return {"message": "API running 🚀"}


@app.post("/predict")
def predict(input_data: InputData):
    data = np.array(input_data.data).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}