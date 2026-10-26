# Actividad 5 - Entrenamiento y Evaluación de Modelos con MLflow

## Descripción del proyecto

Este proyecto tiene como objetivo implementar un flujo de trabajo de Machine Learning que permita preparar datos, entrenar modelos, optimizar hiperparámetros y registrar experimentos utilizando MLflow.

Se utilizará un dataset público para resolver un problema de clasificación binaria, comparando el rendimiento de dos algoritmos de aprendizaje supervisado.

## Dataset seleccionado

- **Nombre:** Breast Cancer Wisconsin (Diagnostic).
- **Fuente:** UCI Machine Learning Repository, disponible mediante Scikit-learn.
- **Registros:** 569.
- **Características predictoras:** 30 variables numéricas.
- **Variable objetivo:** Clasificación binaria (benigno o maligno).

## Algoritmos seleccionados

1. Regresión Logística.
2. Random Forest Classifier.

## Metodología

1. Obtención y almacenamiento del dataset original.
2. Exploración, validación y limpieza de datos.
3. Separación de datos de entrenamiento y prueba.
4. Entrenamiento de dos modelos de clasificación.
5. Optimización de hiperparámetros mediante GridSearchCV.
6. Validación cruzada con Stratified K-Fold.
7. Evaluación mediante Accuracy, Precision, Recall, F1-score y ROC-AUC.
8. Registro de parámetros, métricas y modelos en MLflow.
9. Comparación de resultados y selección del mejor modelo.

## Estructura del repositorio

```text
Actividad5/
├── datos/
│   ├── datos_ini/
│   └── datos_limp/
├── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
├── README.md
└── CHANGELOG.md
```

## Tecnologías utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- MLflow
- GitHub

## Reproducción del proyecto

Las instrucciones detalladas de instalación, preparación de datos y ejecución de experimentos se incorporarán una vez implementados y validados los scripts.

## Estado del proyecto

En desarrollo. Estructura inicial del repositorio creada.
