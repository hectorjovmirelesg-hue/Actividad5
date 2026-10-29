
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
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

    # Configuracion de MLflow
    Path("mlflow_data").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(
        "sqlite:///mlflow_data/mlflow.db"
    )
    mlflow.set_experiment(
        "Actividad5_Comparacion_Modelos"
    )

    # Cargar datos
    df = pd.read_csv(ruta_datos)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Separacion entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Validacion cruzada
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

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

        print(f"\nEntrenando: {nombre}")

        busqueda = GridSearchCV(
            estimator=config["modelo"],
            param_grid=config["parametros"],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True
        )

        busqueda.fit(X_train, y_train)

        modelo = busqueda.best_estimator_
        predicciones = modelo.predict(X_test)
        probabilidades = modelo.predict_proba(X_test)[:, 1]

        metricas = {
            "accuracy": accuracy_score(y_test, predicciones),
            "precision": precision_score(y_test, predicciones),
            "recall": recall_score(y_test, predicciones),
            "f1_score": f1_score(y_test, predicciones),
            "roc_auc": roc_auc_score(y_test, probabilidades),
            "cv_f1": busqueda.best_score_
        }

        resultados[nombre] = metricas

        # Registrar experimento completo
        with mlflow.start_run(
            run_name=f"{nombre}_GridSearch"
        ):

            mlflow.log_param("algoritmo", nombre)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)

            mlflow.log_params(busqueda.best_params_)

            mlflow.log_metrics({
                k: float(v)
                for k, v in metricas.items()
            })

            # Registrar modelo
            if nombre == "Random_Forest":
                mlflow.sklearn.log_model(
                    sk_model=modelo,
                    name="modelo_entrenado",
                    skops_trusted_types=[
                        "sklearn.tree._tree.Tree"
                    ]
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=modelo,
                    name="modelo_entrenado"
                )

        print("Mejores parametros:", busqueda.best_params_)
        print("Metricas:", metricas)
        print("Modelo registrado en MLflow")

    return resultados
