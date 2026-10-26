# Actividad 5 
# Entrenamiento y registro de modelos con MLFlow

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def entrenar_modelos(ruta_datos):
    # Cargar dataset limpio
    df = pd.read_csv(ruta_datos)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Definir modelos
    modelos = {
        "Regresion_Logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("modelo", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    resultados = {}

    # Entrenar y evaluar
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        predicciones = modelo.predict(X_test)
        probabilidades = modelo.predict_proba(X_test)[:, 1]

        metricas = {
            "accuracy": accuracy_score(y_test, predicciones),
            "precision": precision_score(y_test, predicciones),
            "recall": recall_score(y_test, predicciones),
            "f1_score": f1_score(y_test, predicciones),
            "roc_auc": roc_auc_score(y_test, probabilidades)
        }

        resultados[nombre] = metricas

        print(f"\nModelo: {nombre}")
        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

    return resultados
