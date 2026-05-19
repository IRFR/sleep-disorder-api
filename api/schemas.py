"""
Modelos Pydantic para validación de entrada en la API
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class SleepInputData(BaseModel):
    """Esquema de validación para datos de entrada de predicción"""
    
    cognitive_performance_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Puntuación de rendimiento cognitivo (0-100)"
    )
    sleep_quality_score: float = Field(
        ..., 
        ge=1, 
        le=10, 
        description="Puntuación de calidad del sueño (1-10)"
    )
    mental_health_condition: Literal[0, 1, 2, 3] = Field(
        ..., 
        description="Condición de salud mental: 0=Depression, 1=Anxiety, 2=Both, 3=Healthy"
    )
    sleep_duration_hrs: float = Field(
        ..., 
        ge=1, 
        le=12, 
        description="Duración del sueño en horas (1-12)"
    )
    bmi: float = Field(
        ..., 
        ge=10, 
        le=50, 
        description="Índice de masa corporal (10-50)"
    )
    stress_score: float = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Puntuación de estrés (0-10)"
    )
    sleep_latency_mins: float = Field(
        ..., 
        ge=0, 
        le=120, 
        description="Latencia del sueño en minutos (0-120)"
    )
    wake_episodes_per_night: float = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Episodios de despertar por noche (0-10)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "cognitive_performance_score": 75,
                "sleep_quality_score": 6.5,
                "mental_health_condition": 3,
                "sleep_duration_hrs": 7.5,
                "bmi": 24.5,
                "stress_score": 5.0,
                "sleep_latency_mins": 15,
                "wake_episodes_per_night": 1
            }
        }
    }


class PredictionResponse(BaseModel):
    """Respuesta de predicción del modelo"""
    
    prediction: str = Field(..., description="Clase predicha: Severe, Moderate, Mild o Healthy")
    prediction_code: int = Field(..., description="Código numérico de la predicción")
    probabilities: dict = Field(..., description="Probabilidades para cada clase")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")


class HealthCheckResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    model_loaded: bool
    version: str
