"""
Beijing Air Quality Predictor — Backend FastAPI
Entrena los modelos al iniciar si no existen, o carga los guardados.
Ejecutar: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import joblib
import os
from pathlib import Path

# ── Modelos ──────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd


LR_PATH   = MODEL_DIR / "lr_model.pkl"
TREE_PATH = MODEL_DIR / "tree_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Predictores que usa el modelo (sin PM2.5 para evitar data leakage)
PREDICTORS = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP"]


# def train_and_save():
#     """
#     Entrena los modelos con datos sintéticos representativos del dataset de Beijing.
#     Reemplaza esto con tu df_knn y df_clean reales si ya los tienes exportados.
#     """
#     print("⚙️  Entrenando modelos (primera vez)...")

#     import kagglehub, glob
#     path = kagglehub.dataset_download('sid321axn/beijing-multisite-airquality-data-set')
#     archivos = glob.glob(os.path.join(path, '*.csv'))
#     df_raw = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)

#     df = df_raw.copy()
#     df['aire_peligroso'] = (df['PM2.5'] >= 150).astype(int)

#     numeric_cols = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
#     from sklearn.impute import KNNImputer
#     imp = KNNImputer(n_neighbors=5)
#     df[numeric_cols] = imp.fit_transform(df[numeric_cols])

#     # Capping P1-P99
#     for col in numeric_cols:
#         df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

#     X = df[PREDICTORS]
#     y = df['aire_peligroso']

#     from sklearn.model_selection import train_test_split
#     X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train_sc = scaler.fit_transform(X_train)

#     lr = LogisticRegression(max_iter=1000, random_state=42)
#     lr.fit(X_train_sc, y_train)

#     tree = DecisionTreeClassifier(max_depth=6, random_state=42)
#     tree.fit(X_train, y_train)

#     joblib.dump(lr, LR_PATH)
#     joblib.dump(tree, TREE_PATH)
#     joblib.dump(scaler, SCALER_PATH)
#     return lr, tree, scaler


def load_models():
    if LR_PATH.exists() and TREE_PATH.exists() and SCALER_PATH.exists():
        print("📦 Cargando modelos desde disco...")
        return (
            joblib.load(LR_PATH),
            joblib.load(TREE_PATH),
            joblib.load(SCALER_PATH),
        )
    return train_and_save()


lr_model, tree_model, scaler = load_models()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Beijing Air Quality API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción limita a tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    PM10:  float
    SO2:   float
    NO2:   float
    CO:    float
    O3:    float
    TEMP:  float
    # Extras informativos (no entran al modelo)
    PM25:  float = 0.0
    PRES:  float = 1013.0
    DEWP:  float = 0.0
    RAIN:  float = 0.0
    WSPM:  float = 1.5
    station: str = "Dongsi"
    month:   int = 6


class PredictResponse(BaseModel):
    prediction_lr:    int
    prediction_tree:  int
    proba_lr:         float
    proba_tree:       float
    risk_score:       float
    aqi_estimated:    int
    label:            str
    danger:           bool


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models": ["LogisticRegression", "DecisionTreeClassifier"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    features = [[req.PM10, req.SO2, req.NO2, req.CO, req.O3, req.TEMP]]

    # Logistic Regression (requiere escalado)
    features_sc = scaler.transform(features)
    pred_lr    = int(lr_model.predict(features_sc)[0])
    proba_lr   = float(lr_model.predict_proba(features_sc)[0][1])

    # Decision Tree
    pred_tree  = int(tree_model.predict(features)[0])
    proba_tree = float(tree_model.predict_proba(features)[0][1])

    # Risk score combinado (promedio ponderado: 40% LR, 60% Tree)
    risk_score = round(0.4 * proba_lr + 0.6 * proba_tree, 4)

    # AQI estimado simple
    aqi = int(req.PM25 * 0.89 + req.PM10 * 0.08 + req.NO2 * 0.03)

    danger = risk_score >= 0.5 or req.PM25 >= 150

    label = (
        "Aire muy peligroso — evitar exposición"  if risk_score >= 0.75 or req.PM25 >= 150 else
        "Aire peligroso para la salud"            if 0.5  <= risk_score < 0.75 else
        "Aire peligroso para grupos sensibles"    if 0.35 <= risk_score < 0.5 else
        "Calidad del aire aceptable"
    )

    return PredictResponse(
        prediction_lr=pred_lr,
        prediction_tree=pred_tree,
        proba_lr=round(proba_lr, 4),
        proba_tree=round(proba_tree, 4),
        risk_score=risk_score,
        aqi_estimated=aqi,
        label=label,
        danger=danger,
    )


# Servir el frontend estático desde /static
if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
