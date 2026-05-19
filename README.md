# 😴 Sleep Disorder Prediction API

API interactiva para predicción de trastornos de sueño usando Machine Learning (RandomForest). Formulario web con validación en tiempo real usando HTMX.

## 🎯 Características

- ✅ **API REST** con FastAPI
- ✅ **Formulario interactivo** con HTMX (sin recargas)
- ✅ **Modelo ML** entrenado con RandomForest y SMOTE
- ✅ **Validación de datos** con Pydantic
- ✅ **8 features** de entrada con rangos validados
- ✅ **4 categorías** de predicción: Severe, Moderate, Mild, Healthy
- ✅ **Documentación automática** con Swagger/OpenAPI
- ✅ **Dockerizado** para fácil despliegue
- ✅ **Listo para Render** (deployment en 5 min)

## 📋 Requisitos

- **Python 3.11+**
- **pip** (gestor de paquetes Python)
- **Git** (para versionado)
- **Cuenta en Render** (para despliegue, opcional)

## 🚀 Inicio Rápido (Local)

### 1. Clonar/Descargar proyecto

```bash
# Si está en GitHub
git clone <tu-repo>
cd PROYECTO\ TP3

# O simplemente abre la carpeta del proyecto
cd "c:\Users\1337Potato\Documents\CODING\PROYECTO TP3"
```

### 2. Crear entorno virtual

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux (bash)
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrenar y serializar el modelo

```bash
python train_model.py
```

**Output esperado:**
```
Cargando dataset...
Preparando datos y aplicando SMOTE...
Buscando parámetro óptimo n_estimators...
  n_estimators=10: error=0.1234
  ...
  n_estimators=500: error=0.0856

--- Modelo entrenado con éxito ---
Número de árboles utilizado: 100
Precisión final del modelo: 92.34%

Serializando modelo...
✓ Modelo guardado: sleep_model.pkl
✓ Metadatos guardados: model_metadata.pkl

✓ Listo para usar en la API!
```

### 5. Ejecutar la API

```bash
uvicorn api.main:app --reload
```

**Output esperado:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
✓ API iniciada correctamente
✓ Modelo cargado: True
  - Precisión del modelo: 92.34%
  - Features: 8 features
```

### 6. Abrir en navegador

- **Formulario web:** http://localhost:8000/
- **Documentación API:** http://localhost:8000/docs (Swagger)
- **Health check:** http://localhost:8000/health

## 📝 Uso de la API

### Endpoint POST `/predict`

**Request (JSON):**
```json
{
  "cognitive_performance_score": 75,
  "sleep_quality_score": 6.5,
  "mental_health_condition": 3,
  "sleep_duration_hrs": 7.5,
  "bmi": 24.5,
  "stress_score": 5.0,
  "sleep_latency_mins": 15,
  "wake_episodes_per_night": 1
}
```

**Response:**
```json
{
  "prediction": "Healthy",
  "prediction_code": 3,
  "probabilities": {
    "Severe": 0.02,
    "Moderate": 0.05,
    "Mild": 0.15,
    "Healthy": 0.78
  },
  "confidence": 0.78
}
```

### Parámetros de entrada

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| `cognitive_performance_score` | 0-100 | Puntuación de rendimiento cognitivo |
| `sleep_quality_score` | 1-10 | Calidad subjetiva del sueño |
| `mental_health_condition` | 0-3 | 0=Depression, 1=Anxiety, 2=Both, 3=Healthy |
| `sleep_duration_hrs` | 1-12 | Horas de sueño por noche |
| `bmi` | 10-50 | Índice de masa corporal |
| `stress_score` | 0-10 | Nivel de estrés |
| `sleep_latency_mins` | 0-120 | Minutos para conciliar el sueño |
| `wake_episodes_per_night` | 0-10 | Veces que se despierta por noche |

## 🐳 Despliegue Local con Docker

```bash
# Construir imagen Docker
docker build -t sleep-api .

# Ejecutar contenedor
docker run -p 8000:8000 sleep-api

# Acceder en http://localhost:8000
```

## ☁️ Despliegue en Render (Recomendado)

### Opción A: Desde GitHub (Automático)

1. **Subir proyecto a GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Sleep Disorder Prediction API"
   git branch -M main
   git remote add origin https://github.com/tu-usuario/sleep-api.git
   git push -u origin main
   ```

2. **Conectar a Render**
   - Ir a https://render.com/
   - Click en "New +" → "Web Service"
   - Conectar repositorio GitHub
   - Seleccionar rama `main`
   - Render leerá automáticamente `render.yaml`
   - Click en "Create Web Service"

3. **Esperar a que se despliegue** (~2-3 min)

   Tu API estará disponible en: `https://sleep-disorder-api.onrender.com`

### Opción B: Desde Docker (Manual)

1. Crear cuenta en [Render.com](https://render.com/)
2. Click en "New +" → "Web Service"
3. Seleccionar "Docker" como entorno
4. Usar imagen: `ghcr.io/tu-usuario/sleep-api:latest` (si la pusheaste a GitHub Container Registry)

## 📊 Estructura del Proyecto

```
PROYECTO TP3/
├── api/
│   ├── __init__.py
│   ├── main.py              # Aplicación FastAPI
│   ├── schemas.py           # Modelos Pydantic
│   └── models.py            # Gestor del modelo ML
├── templates/
│   └── index.html           # Formulario HTMX interactivo
├── train_model.py           # Script de entrenamiento
├── requirements.txt         # Dependencias Python
├── Dockerfile              # Configuración Docker
├── render.yaml             # Configuración Render
├── .gitignore              # Archivos ignorados por Git
└── README.md               # Este archivo
```

## 🔍 Troubleshooting

### Error: "Modelo no encontrado en sleep_model.pkl"

**Solución:** Ejecutar primero `python train_model.py`

```bash
python train_model.py
```

### Error: ModuleNotFoundError: No module named 'fastapi'

**Solución:** Instalar dependencias
```bash
pip install -r requirements.txt
```

### Puerto 8000 ya está en uso

**Solución:** Usar otro puerto
```bash
uvicorn api.main:app --port 8001
```

### ¿Cómo cambiar de despliegue local a Render?

No necesitas cambiar nada. El código es portable. Solo:
1. Sube a GitHub
2. Conecta a Render
3. ¡Listo!

## 📚 Endpoints Disponibles

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Formulario HTML |
| `POST` | `/predict` | Predicción JSON |
| `POST` | `/predict-form` | Predicción desde formulario HTML |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Información del modelo |
| `GET` | `/docs` | Documentación Swagger |
| `GET` | `/redoc` | Documentación ReDoc |

## 🛠️ Desarrollo

### Modificar el modelo

Editar `train_model.py` para cambiar:
- Features a usar
- Parámetros de SMOTE
- Hiperparámetros de RandomForest
- Test/Train split

Luego ejecutar:
```bash
python train_model.py
```

### Agregar nuevos endpoints

Editar `api/main.py` y agregar rutas con decoradores `@app.get()` o `@app.post()`

### Personalizar formulario

Editar `templates/index.html` para cambiar:
- Campos del formulario
- Estilos CSS
- Textos y etiquetas

## 📄 Licencia

Proyecto de demostración con fines educativos.

## 👤 Autor

Creado como parte del Proyecto TP3 - Predicción de Trastornos de Sueño

## 🤝 Soporte

Si encuentras problemas:
1. Revisa la sección Troubleshooting
2. Verifica que las dependencias estén instaladas: `pip list`
3. Asegúrate de que `sleep_model.pkl` existe
4. Intenta desde una carpeta limpia si persisten los errores

---

**¡Listo para predecir trastornos de sueño!** 🌙
