#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para entrenar y serializar el modelo RandomForest de predicción de trastornos de sueño.

Uso:
  python train_model.py

Genera:
  - sleep_model.pkl (modelo entrenado)
  - model_metadata.pkl (metadatos del modelo)
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import glob
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("\n" + "="*70)
print("SLEEP DISORDER PREDICTION - MODEL TRAINING")
print("="*70)

try:
    # ─── 1. CARGA DEL DATASET ─────────────────────────────────────────────
    print("\n[1/6] Cargando dataset...")
    print(f"  Directorio actual: {os.getcwd()}")
    
    # Buscar el archivo CSV (puede tener caracteres especiales)
    csv_files = glob.glob('*.csv') + glob.glob('*/*.csv')
    print(f"  Archivos CSV encontrados: {csv_files}")
    
    csv_file = None
    for f in csv_files:
        if 'Salud' in f or 'sueno' in f.lower() or 'sueño' in f.lower():
            csv_file = f
            print(f"  ✓ Usando dataset: {csv_file}")
            break
    
    if not csv_file:
        raise FileNotFoundError(f"No se encontró CSV en {os.getcwd()}\nArchivos: {os.listdir('.')}")
    
    dataframe = pd.read_csv(csv_file, sep=';')
    print(f"  ✓ Dataset cargado: {dataframe.shape[0]} filas, {dataframe.shape[1]} columnas")
    
    # ─── 2. DEFINICIÓN DE FEATURES ─────────────────────────────────────────
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
    
    # ─── 3. PREPARACIÓN DE DATOS Y BALANCEO ────────────────────────────────
    print("\n[3/6] Preparando datos y aplicando SMOTE...")
    x_top8_raw = dataframe[top8_features]
    y_top8_raw = dataframe['sleep_disorder_risk']
    print(f"  Data shape: {x_top8_raw.shape}")
    print(f"  Clases únicas: {sorted(y_top8_raw.unique())}")
    
    smote_top8 = SMOTE(random_state=42)
    x_bal, y_bal = smote_top8.fit_resample(x_top8_raw, y_top8_raw)
    print(f"  ✓ SMOTE aplicado: {x_bal.shape}")
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_bal, y_bal, test_size=0.2, random_state=30, stratify=y_bal
    )
    print(f"  ✓ Train/Test split: {x_train.shape} / {x_test.shape}")
    
    # ─── 4. BÚSQUEDA DEL PARÁMETRO ÓPTIMO ──────────────────────────────────
    print("\n[4/6] Buscando parámetro óptimo n_estimators...")
    rango_arboles = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
    mejor_n = 100
    mejor_error = float('inf')
    
    for n in rango_arboles:
        rf_i = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        rf_i.fit(x_train, y_train)
        pred_i = rf_i.predict(x_test)
        error = np.mean(pred_i != y_test)
        print(f"  n_estimators={n}: error={error:.4f}")
        if error < mejor_error:
            mejor_error = error
            mejor_n = n
    
    print(f"  ✓ Parámetro óptimo: {mejor_n} árboles (error: {mejor_error:.4f})")
    
    # ─── 5. ENTRENAMIENTO DEL MODELO FINAL ─────────────────────────────────
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
    
    # ─── 6. VALIDACIÓN Y SERIALIZACIÓN ─────────────────────────────────────
    print("\n[6/6] Evaluando y guardando modelo...")
    predicciones = modelo_final.predict(x_test)
    precision = accuracy_score(y_test, predicciones)
    print(f"  Precisión: {precision*100:.2f}%")
    
    # Guardar modelo
    joblib.dump(modelo_final, 'sleep_model.pkl')
    print(f"  ✓ Modelo guardado: sleep_model.pkl")
    
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
    joblib.dump(metadata, 'model_metadata.pkl')
    print(f"  ✓ Metadatos guardados: model_metadata.pkl")
    
    print("\n" + "="*70)
    print("✓ ENTRENAMIENTO COMPLETADO CON ÉXITO")
    print("="*70 + "\n")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
