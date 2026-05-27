# Beijing Air Quality Predictor

Dashboard interactivo para predecir la calidad del aire en Beijing usando Regresión Logística y Árbol de Decisión, entrenados con el dataset multiestación 2013–2017 (420,768 registros, 12 estaciones de monitoreo).

---

## Descripción del proyecto

El modelo predice si el aire es peligroso para la salud (PM2.5 >= 150 µg/m³) a partir de seis variables: PM10, SO2, NO2, CO, O3 y temperatura. Se entrenaron dos clasificadores con el pipeline completo del notebook original: imputación KNN, capping P1-P99, escalado con StandardScaler y split 80/20 estratificado.

El backend expone una API REST en FastAPI. El frontend es un archivo HTML estático con sliders para cada variable, gráfico comparativo contra umbrales EPA y visualización del score combinado (40% Regresión Logística + 60% Árbol de Decisión).

---

## Estructura del repositorio

```
beijing-air-dashboard/
├── app.py                  Backend FastAPI
├── requirements.txt
├── .python-version         Fija Python 3.11.9 para el deploy
├── vercel.json             Configuracion para Vercel (solo frontend)
├── models/
│   ├── lr_model.pkl        Regresion Logistica entrenada
│   ├── tree_model.pkl      Arbol de Decision entrenado
│   └── scaler.pkl          StandardScaler ajustado al conjunto de entrenamiento
└── static/
    └── index.html          Dashboard (se sirve como archivo estatico)
```

---

## Requisitos

- Python 3.11
- Las dependencias están en `requirements.txt`

---

## Ejecución local

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Abrir en el navegador: `http://localhost:8000`

Para verificar que la API responde:

```bash
curl http://localhost:8000/health
```

Respuesta esperada:

```json
{"status": "ok", "models": ["LogisticRegression", "DecisionTreeClassifier"]}
```

---

## Endpoint de prediccion

`POST /predict`

Cuerpo de la peticion:

```json
{
  "PM10": 120,
  "SO2": 20,
  "NO2": 55,
  "CO": 1200,
  "O3": 60,
  "TEMP": 12,
  "PM25": 85
}
```

Respuesta:

```json
{
  "prediction_lr": 0,
  "prediction_tree": 0,
  "proba_lr": 0.21,
  "proba_tree": 0.18,
  "risk_score": 0.194,
  "aqi_estimated": 87,
  "label": "Calidad del aire aceptable",
  "danger": false
}
```

---

## Actualizar los modelos

Si se reentrenan los modelos en el notebook, exportarlos con:

```python
import joblib

joblib.dump(log_reg,      "models/lr_model.pkl")
joblib.dump(tree_clf,     "models/tree_model.pkl")
joblib.dump(scaler_model, "models/scaler.pkl")
```

Luego hacer push al repositorio. Render redespliega automaticamente.

---

## Deploy en Render

El proyecto esta desplegado en Render con auto-deploy activado desde la rama `main`.

Configuracion del servicio:

| Campo | Valor |
|---|---|
| Runtime | Python 3.11 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |

Cada `git push` a `main` dispara un nuevo deploy automaticamente.

URL de produccion: `https://beijing-air-dashboard.onrender.com`

---

## Variables del modelo

| Variable | Descripcion | Unidad | Rango dataset |
|---|---|---|---|
| PM10 | Material particulado grueso | µg/m³ | 0 – 600 |
| SO2 | Dioxido de azufre | µg/m³ | 0 – 200 |
| NO2 | Dioxido de nitrogeno | µg/m³ | 0 – 250 |
| CO | Monoxido de carbono | µg/m³ | 0 – 10000 |
| O3 | Ozono | µg/m³ | 0 – 300 |
| TEMP | Temperatura | °C | -20 – 42 |

Variable objetivo: `aire_peligroso` = 1 si PM2.5 >= 150 µg/m³, 0 en caso contrario.

---

## Fuente de datos

Dataset: [Beijing Multi-Site Air Quality](https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set)  
Periodo: Marzo 2013 – Febrero 2017  
Estaciones: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong
