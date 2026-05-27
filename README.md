# Beijing Air Quality Predictor — Guía de Deploy

## Estructura del proyecto

```
beijing-air-dashboard/
├── app.py              ← Backend FastAPI
├── requirements.txt
├── models/             ← Se genera automáticamente al primer arranque
│   ├── lr_model.pkl
│   ├── tree_model.pkl
│   └── scaler.pkl
└── static/
    └── index.html      ← Dashboard (se sirve desde FastAPI)
```

---

## Ejecutar localmente

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Arrancar
uvicorn app:app --reload --port 8000
```

Abre: http://localhost:8000

---

## Usar tus modelos reales del notebook

En `app.py`, reemplaza la función `train_and_save()` con:

```python
def train_and_save():
    import kagglehub, glob
    path = kagglehub.dataset_download('sid321axn/beijing-multisite-airquality-data-set')
    archivos = glob.glob(os.path.join(path, '*.csv'))
    df_raw = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)

    df = df_raw.copy()
    df['aire_peligroso'] = (df['PM2.5'] >= 150).astype(int)

    numeric_cols = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']
    from sklearn.impute import KNNImputer
    imp = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imp.fit_transform(df[numeric_cols])

    # Capping P1-P99
    for col in numeric_cols:
        df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

    X = df[PREDICTORS]
    y = df['aire_peligroso']

    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)

    tree = DecisionTreeClassifier(max_depth=6, random_state=42)
    tree.fit(X_train, y_train)

    joblib.dump(lr, LR_PATH)
    joblib.dump(tree, TREE_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return lr, tree, scaler
```

O si ya tienes los `.pkl` del notebook, simplemente cópialos en `./models/`.

---

## Deploy en Render (recomendado, gratis)

1. Sube el proyecto a GitHub
2. Ve a https://render.com → New Web Service
3. Conecta tu repo
4. Configura:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. En el dashboard HTML, cambia la API URL a `https://tu-app.onrender.com`

---

## Deploy en Railway

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

---

## Deploy en VPS (Ubuntu)

```bash
# En el servidor
git clone https://github.com/tu-user/beijing-air-dashboard
cd beijing-air-dashboard
pip install -r requirements.txt

# Instalar PM2 o usar systemd
pip install gunicorn
gunicorn app:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Con nginx como reverse proxy:
# location / { proxy_pass http://127.0.0.1:8000; }
```

---

## CORS en producción

Cambia en `app.py`:
```python
allow_origins=["https://tudominio.com"]
```
