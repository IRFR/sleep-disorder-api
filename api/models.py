"""
Funciones para cargar y gestionar el modelo ML
"""

import joblib
import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ModelManager:
    """Gestor centralizado del modelo ML con capacidad de auto-entrenamiento"""
    
    _instance = None
    _model = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            # Intenta cargar modelo existente
            self.load_model()
    
    def get_model_paths(self):
        """Determina rutas correctas del modelo"""
        if os.path.exists('sleep_model.pkl'):
            return 'sleep_model.pkl', 'model_metadata.pkl'
        else:
            base_path = Path(__file__).parent.parent
            return str(base_path / 'sleep_model.pkl'), str(base_path / 'model_metadata.pkl')
    
    def load_model(self):
        """Carga el modelo serializado y sus metadatos"""
        model_path, metadata_path = self.get_model_paths()
        
        try:
            if not os.path.exists(model_path):
                print(f"⚠️  Archivo de modelo no encontrado en: {model_path}")
                return False
            
            self._model = joblib.load(model_path)
            self._metadata = joblib.load(metadata_path)
            print(f"✓ Modelo cargado correctamente desde {model_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error al cargar el modelo: {e}")
            return False
    
    def train_model_if_missing(self):
        """Entrena el modelo automáticamente si no existe (para Render deployment)"""
        model_path, _ = self.get_model_paths()
        
        if os.path.exists(model_path):
            print("✓ Modelo ya existe, cargando...")
            self.load_model()
            return True
        
        print("\n" + "="*70)
        print("AUTO-TRAINING MODEL (No model file found)")
        print("="*70)
        
        try:
            # 1. CARGA DEL DATASET
            print("\n[1/6] Buscando dataset...")
            csv_files = glob.glob('*.csv') + glob.glob('*/*.csv')
            print(f"  CSV encontrados: {csv_files}")
            
            csv_file = None
            for f in csv_files:
                if 'Salud' in f or 'sueno' in f.lower() or 'sueño' in f.lower():
                    csv_file = f
                    break
            
            if not csv_file:
                print(f"  ✗ No se encontró CSV en {os.getcwd()}")
                print(f"  Archivos disponibles: {os.listdir('.')}")
                return False
            
            print(f"  ✓ Cargando: {csv_file}")
            dataframe = pd.read_csv(csv_file, sep=';')
            print(f"  ✓ Dataset: {dataframe.shape[0]} filas, {dataframe.shape[1]} columnas")
            
            # 2. DEFINICIÓN DE FEATURES
            print("\n[2/6] Definiendo features...")
            top8_features = [
                'cognitive_performance_score',
                'sleep_quality_score',
                'mental_health_condition',
                'sleep_duration_hrs',
                'bmi',
                'stress_score',
                'sleep_latency_mins',
                'wake_episodes_per_night',
            ]
            print(f"  ✓ {len(top8_features)} features seleccionados")
            
            # 3. PREPARACIÓN DE DATOS
            print("\n[3/6] Preparando datos...")
            x_top8_raw = dataframe[top8_features]
            y_top8_raw = dataframe['sleep_disorder_risk']
            
            smote = SMOTE(random_state=42)
            x_bal, y_bal = smote.fit_resample(x_top8_raw, y_top8_raw)
            print(f"  ✓ SMOTE aplicado: {x_bal.shape}")
            
            x_train, x_test, y_train, y_test = train_test_split(
                x_bal, y_bal, test_size=0.2, random_state=30, stratify=y_bal
            )
            print(f"  ✓ Train/Test split: {x_train.shape} / {x_test.shape}")
            
            # 4. BÚSQUEDA DEL PARÁMETRO ÓPTIMO
            print("\n[4/6] Optimizando n_estimators...")
            rango_arboles = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
            mejor_n = 100
            mejor_error = float('inf')
            
            for n in rango_arboles:
                rf_i = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
                rf_i.fit(x_train, y_train)
                pred_i = rf_i.predict(x_test)
                error = np.mean(pred_i != y_test)
                if error < mejor_error:
                    mejor_error = error
                    mejor_n = n
            
            print(f"  ✓ Parámetro óptimo: {mejor_n} árboles")
            
            # 5. ENTRENAMIENTO FINAL
            print("\n[5/6] Entrenando modelo final...")
            modelo_final = RandomForestClassifier(
                n_estimators=mejor_n,
                max_depth=None,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            modelo_final.fit(x_train, y_train)
            print(f"  ✓ Modelo entrenado")
            
            # 6. VALIDACIÓN Y SERIALIZACIÓN
            print("\n[6/6] Evaluando y guardando...")
            predicciones = modelo_final.predict(x_test)
            precision = accuracy_score(y_test, predicciones)
            print(f"  Precisión: {precision*100:.2f}%")
            
            # Guardar modelo
            joblib.dump(modelo_final, model_path)
            print(f"  ✓ Modelo guardado: {model_path}")
            
            # Guardar metadatos
            metadata = {
                'features': top8_features,
                'classes': list(modelo_final.classes_),
                'class_names': {
                    0: 'Severe',
                    1: 'Moderate',
                    2: 'Mild',
                    3: 'Healthy'
                },
                'n_estimators': mejor_n,
                'accuracy': precision
            }
            joblib.dump(metadata, self.get_model_paths()[1])
            print(f"  ✓ Metadatos guardados")
            
            # Cargar en memoria
            self._model = modelo_final
            self._metadata = metadata
            
            print("\n" + "="*70)
            print("✓ MODELO ENTRENADO Y GUARDADO EXITOSAMENTE")
            print("="*70 + "\n")
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR durante auto-training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, input_data: dict) -> dict:
        """Realiza una predicción con el modelo"""
        if self._model is None:
            print("⚠️  Modelo no cargado, intentando cargar...")
            if not self.load_model():
                raise RuntimeError("Modelo no disponible para predicción")
        
        # Extraer features en orden correcto
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
