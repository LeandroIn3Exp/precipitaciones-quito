import os
import joblib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import numpy as np

# Cargar modelo y mapeo de estaciones
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_results')
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_lluvia.pkl')
MAPPING_PATH = os.path.join(MODEL_DIR, 'station_mapping.json')

best_model = joblib.load(MODEL_PATH)
with open(MAPPING_PATH, 'r') as f:
    station_mapping = json.load(f)

app = FastAPI()

class PredictRequest(BaseModel):
    station: str
    date: str  # formato YYYY-MM-DD

class PredictResponse(BaseModel):
    station: str
    date: str
    probabilidad_lluvia: float
    llovera: bool

@app.post("/predict", response_model=PredictResponse)
def predict_lluvia(req: PredictRequest):
    # Validar estación
    if req.station not in station_mapping:
        raise HTTPException(status_code=404, detail=f"Estación '{req.station}' no encontrada. Usa una de: {list(station_mapping.keys())}")
    # Validar fecha
    try:
        fecha = datetime.strptime(req.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Fecha debe tener formato YYYY-MM-DD")
    # Extraer features
    features = {
        'station_code': station_mapping[req.station],
        'año': fecha.year,
        'mes': fecha.month,
        'dia': fecha.day,
        'dia_del_año': fecha.timetuple().tm_yday,
        'dia_de_la_semana': fecha.weekday(),
        'semana_del_año': int(fecha.strftime('%V'))
    }
    X = np.array([[features['station_code'], features['año'], features['mes'], features['dia'],
                   features['dia_del_año'], features['dia_de_la_semana'], features['semana_del_año']]])
    proba = best_model.predict_proba(X)[0,1]
    pred = bool(best_model.predict(X)[0])
    return PredictResponse(
        station=req.station,
        date=req.date,
        probabilidad_lluvia=round(proba, 4),
        llovera=pred
    ) 