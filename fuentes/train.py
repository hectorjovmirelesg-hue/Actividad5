
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
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

    # Mantener la misma division inicial
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Validacion cruzada estratificada
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Modelos e hiperparametros
    configuraciones = {
        "Regresion_Logistica": {
            "modelo": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", LogisticRegression(
                    max_iter=2000,
                    random_state=42
                ))
            ]),
            "parametros": {
                "modelo__C": [0.01, 0.1, 1, 10, 100]
            }
        },
        "Random_Forest": {
            "modelo": RandomForestClassifier(
                random_state=42
            ),
            "parametros": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }
        }
    }

    resultados = {}

    for nombre, config in configuraciones.items():
        print(f"\nOptimizando: {nombre}")

        busqueda = GridSearchCV(
            estimator=config["modelo"],
            param_grid=config["parametros"],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True
        )

        # Ajustar exclusivamente con entrenamiento
        busqueda.fit(X_train, y_train)

        mejor_modelo = busqueda.best_estimator_

        # Evaluacion final con datos de prueba
        predicciones = mejor_modelo.predict(X_test)
        probabilidades = mejor_modelo.predict_proba(X_test)[:, 1]

        metricas = {
            "accuracy": accuracy_score(y_test, predicciones),
            "precision": precision_score(y_test, predicciones),
            "recall": recall_score(y_test, predicciones),
            "f1_score": f1_score(y_test, predicciones),
            "roc_auc": roc_auc_score(y_test, probabilidades),
            "cv_f1": busqueda.best_score_
        }

        resultados[nombre] = metricas

        print("Mejores hiperparametros:")
        print(busqueda.best_params_)
        print(f"F1 promedio CV: {busqueda.best_score_:.4f}")

        print("Metricas de prueba:")
        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

    return resultados
