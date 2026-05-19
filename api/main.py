"""
API FastAPI para predicción de trastornos de sueño
Formulario interactivo con HTMX para predecir riesgo de trastornos de sueño
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

from api.schemas import SleepInputData, PredictionResponse, HealthCheckResponse
from api.models import model_manager

# ─── INICIALIZACIÓN DE LA APLICACIÓN ────────────────────────────────────────
app = FastAPI(
    title="Sleep Disorder Prediction API",
    description="API para predicción de trastornos de sueño usando RandomForest",
    version="1.0.0",
)

# Configurar directorio de templates
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"


# ─── RUTAS ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_form():
    """Devuelve el formulario HTML interactivo"""
    with open(TEMPLATES_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(data: SleepInputData) -> PredictionResponse:
    """
    Realiza una predicción con los datos proporcionados.
    
    Ejemplo de entrada:
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
    """
    try:
        # Convertir input a dict
        input_dict = data.model_dump()
        
        # Realizar predicción con el modelo
        result = model_manager.predict(input_dict)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        )


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request):
    """
    Endpoint alternativo para formularios HTML (usado por HTMX).
    Retorna HTML con el resultado en lugar de JSON.
    """
    try:
        # Parsear datos del formulario
        form_data = await request.form()
        
        # Convertir a SleepInputData para validación
        data = SleepInputData(
            cognitive_performance_score=float(form_data['cognitive_performance_score']),
            sleep_quality_score=float(form_data['sleep_quality_score']),
            mental_health_condition=int(form_data['mental_health_condition']),
            sleep_duration_hrs=float(form_data['sleep_duration_hrs']),
            bmi=float(form_data['bmi']),
            stress_score=float(form_data['stress_score']),
            sleep_latency_mins=float(form_data['sleep_latency_mins']),
            wake_episodes_per_night=float(form_data['wake_episodes_per_night']),
        )
        
        # Realizar predicción
        input_dict = data.model_dump()
        result = model_manager.predict(input_dict)
        
        # Generar HTML de resultado
        html = f"""
        <div id="result-container" hx-swap="outerHTML">
            <div class="result-box success-box">
                <h2>📊 Predicción Realizada</h2>
                
                <div class="prediction-main">
                    <p class="prediction-label">Riesgo de Trastorno de Sueño:</p>
                    <p class="prediction-value {result['prediction'].lower()}">{result['prediction'].upper()}</p>
                    <p class="confidence">Confianza: {result['confidence']*100:.1f}%</p>
                </div>
                
                <div class="probabilities">
                    <p class="prob-title">Probabilidades por categoría:</p>
                    <table class="prob-table">
                        <tbody>
        """
        
        for class_name, prob in result['probabilities'].items():
            percentage = prob * 100
            bar_length = int(percentage / 5)  # Barra visual simple
            bar = "█" * bar_length + "░" * (20 - bar_length)
            html += f"""
                            <tr>
                                <td class="prob-name">{class_name}</td>
                                <td class="prob-bar">{bar}</td>
                                <td class="prob-percent">{percentage:.1f}%</td>
                            </tr>
            """
        
        html += """
                        </tbody>
                    </table>
                </div>
                
                <button class="btn-secondary" hx-get="/" hx-target="body" hx-swap="innerHTML">
                    🔄 Nueva Predicción
                </button>
            </div>
        </div>
        """
        
        return html
    
    except ValueError as e:
        return f"""
        <div id="result-container" hx-swap="outerHTML">
            <div class="result-box error-box">
                <h2>⚠️ Error de Validación</h2>
                <p>{str(e)}</p>
                <button class="btn-secondary" hx-get="/" hx-target="body" hx-swap="innerHTML">
                    🔄 Volver al Formulario
                </button>
            </div>
        </div>
        """
    except Exception as e:
        return f"""
        <div id="result-container" hx-swap="outerHTML">
            <div class="result-box error-box">
                <h2>❌ Error del Servidor</h2>
                <p>Error al procesar predicción: {str(e)}</p>
                <button class="btn-secondary" hx-get="/" hx-target="body" hx-swap="innerHTML">
                    🔄 Volver al Formulario
                </button>
            </div>
        </div>
        """


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint para monitoreo"""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded(),
        version="1.0.0"
    )


@app.get("/model-info")
async def model_info():
    """Retorna información sobre el modelo"""
    metadata = model_manager.get_metadata()
    return {
        "status": "ok",
        "model_info": {
            "n_estimators": metadata.get('n_estimators'),
            "accuracy": metadata.get('accuracy'),
            "features": metadata.get('features'),
            "classes": metadata.get('classes'),
            "class_names": metadata.get('class_names')
        }
    }


# ─── MANEJO DE ERRORES ───────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )


# ─── EVENTO DE INICIO ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("✓ API iniciada correctamente")
    print(f"✓ Modelo cargado: {model_manager.is_loaded()}")
    if model_manager.is_loaded():
        metadata = model_manager.get_metadata()
        print(f"  - Precisión del modelo: {metadata.get('accuracy')*100:.2f}%")
        print(f"  - Features: {len(metadata.get('features', []))} features")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
