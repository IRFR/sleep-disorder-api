"""
Funciones para cargar y gestionar el modelo ML
"""

import joblib
import os
from pathlib import Path


class ModelManager:
    """Gestor centralizado del modelo ML"""
    
    _instance = None
    _model = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self.load_model()
    
    def load_model(self):
        """Carga el modelo serializado y sus metadatos"""
        
        # Determinar la ruta base (directorio padre de api/)
        if os.path.exists('sleep_model.pkl'):
            model_path = 'sleep_model.pkl'
            metadata_path = 'model_metadata.pkl'
        else:
            # Si estamos dentro de /api, subir un nivel
            base_path = Path(__file__).parent.parent
            model_path = base_path / 'sleep_model.pkl'
            metadata_path = base_path / 'model_metadata.pkl'
        
        try:
            if not os.path.exists(str(model_path)):
                raise FileNotFoundError(
                    f"Modelo no encontrado en {model_path}\n"
                    "Ejecuta 'python train_model.py' primero."
                )
            
            self._model = joblib.load(model_path)
            self._metadata = joblib.load(metadata_path)
            print(f"✓ Modelo cargado correctamente desde {model_path}")
            
        except Exception as e:
            print(f"✗ Error al cargar el modelo: {e}")
            raise
    
    def predict(self, input_data: dict) -> dict:
        """
        Realiza una predicción con el modelo
        
        Args:
            input_data: dict con los 8 features en el orden correcto
        
        Returns:
            dict con predicción, probabilidades y confianza
        """
        if self._model is None:
            raise RuntimeError("Modelo no está cargado")
        
        # Extraer features en el orden correcto
        features_list = [
            input_data['cognitive_performance_score'],
            input_data['sleep_quality_score'],
            input_data['mental_health_condition'],
            input_data['sleep_duration_hrs'],
            input_data['bmi'],
            input_data['stress_score'],
            input_data['sleep_latency_mins'],
            input_data['wake_episodes_per_night'],
        ]
        
        # Realizar predicción
        prediction_code = self._model.predict([features_list])[0]
        probabilities_array = self._model.predict_proba([features_list])[0]
        
        # Construir diccionario de probabilidades
        class_names = self._metadata['class_names']
        probabilities = {
            class_names[i]: float(prob)
            for i, prob in enumerate(probabilities_array)
        }
        
        # Confianza es la probabilidad máxima
        confidence = float(probabilities_array.max())
        
        return {
            'prediction': class_names[prediction_code],
            'prediction_code': int(prediction_code),
            'probabilities': probabilities,
            'confidence': confidence
        }
    
    def get_metadata(self) -> dict:
        """Retorna los metadatos del modelo"""
        return self._metadata.copy() if self._metadata else {}
    
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado"""
        return self._model is not None


# Singleton global del modelo
model_manager = ModelManager()
