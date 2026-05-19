import joblib
try:
    model = joblib.load('sleep_model.pkl')
    meta = joblib.load('model_metadata.pkl')
    print(f'✓ Modelo cargado: {type(model).__name__}')
    print(f'✓ Precisión: {meta["accuracy"]*100:.2f}%')
    print(f'✓ Features: {len(meta["features"])} features')
except Exception as e:
    print(f'✗ Error: {e}')