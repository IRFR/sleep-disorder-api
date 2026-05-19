"""
Script para entrenar y serializar el modelo RandomForest de predicción de trastornos de sueño.
Ejecutar una sola vez para generar sleep_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── 1. CARGA DEL DATASET ────────────────────────────────────────────────────
print("Cargando dataset...")
dataframe = pd.read_csv('SaludDelSueñoYRendimientoDiario(1).csv', sep=';')

# ─── 2. DEFINICIÓN DE LAS 8 FEATURES MÁS IMPORTANTES ──────────────────────
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

# ─── 3. PREPARACIÓN DE DATOS Y BALANCEO (SOLO TOP-8) ─────────────────────
print("Preparando datos y aplicando SMOTE...")
x_top8_raw = dataframe[top8_features]
y_top8_raw = dataframe['sleep_disorder_risk']

# Aplicamos SMOTE para balancear clases
smote_top8 = SMOTE(random_state=42)
x_bal, y_bal = smote_top8.fit_resample(x_top8_raw, y_top8_raw)

# División de datos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    x_bal, y_bal, test_size=0.2, random_state=30, stratify=y_bal
)

# ─── 4. BÚSQUEDA DEL PARÁMETRO ÓPTIMO (n_estimators) ──────────────────────
print("Buscando parámetro óptimo n_estimators...")
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
    print(f"  n_estimators={n}: error={error:.4f}")

# ─── 5. ENTRENAMIENTO DEL MODELO FINAL ──────────────────────────────────────
print(f"\nEntrenando modelo final con n_estimators={mejor_n}...")
modelo_final = RandomForestClassifier(
    n_estimators=mejor_n,
    max_depth=None,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
modelo_final.fit(x_train, y_train)

# ─── 6. VALIDACIÓN DEL MODELO ───────────────────────────────────────────────
predicciones = modelo_final.predict(x_test)
precision = accuracy_score(y_test, predicciones)

print(f"\n--- Modelo entrenado con éxito ---")
print(f"Número de árboles utilizado: {mejor_n}")
print(f"Precisión final del modelo: {precision*100:.2f}%")

# ─── 7. SERIALIZACIÓN DEL MODELO Y METADATOS ────────────────────────────────
print("\nSerializando modelo...")

# Guardar el modelo
joblib.dump(modelo_final, 'sleep_model.pkl')
print("✓ Modelo guardado: sleep_model.pkl")

# Guardar metadatos necesarios para la API
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

joblib.dump(metadata, 'model_metadata.pkl')
print("✓ Metadatos guardados: model_metadata.pkl")

print("\n✓ Listo para usar en la API!")
