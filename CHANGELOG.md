# CHANGELOG.md — Actividad 5

## Versión 1.0.0 — Proyecto de Machine Learning con MLflow

### 1. Descripción del proyecto

Se desarrolló un proyecto de aprendizaje automático utilizando un dataset público de clasificación binaria, con el propósito de implementar la preparación de datos, entrenamiento, optimización de hiperparámetros, evaluación y registro de experimentos mediante MLflow.

### 2. Dataset utilizado

**Breast Cancer Wisconsin (Diagnostic)**, disponible en Scikit-learn y procedente de UCI Machine Learning Repository.

- Registros originales: 569.
- Variables predictoras: 30.
- Variable objetivo: `target`.
- Tipo de problema: clasificación binaria.
- Clases: 0 = maligno y 1 = benigno.

Los datos originales y procesados se almacenaron por separado para conservar su trazabilidad.

### 3. Preparación de datos

Se implementó el archivo `datos_prep.py` para:

- Verificar valores nulos.
- Identificar registros duplicados.
- Validar tipos de datos.
- Comprobar los valores de la variable objetivo.
- Generar el dataset limpio.

La separación de datos se realizó utilizando 80% para entrenamiento y 20% para prueba, con estratificación y semilla aleatoria 42.

### 4. Entrenamiento y optimización

Se compararon dos algoritmos:

**Regresión logística**
- Escalado de variables mediante StandardScaler.
- Optimización del parámetro de regularización `C`.
- Mejor valor encontrado: `C = 0.1`.

**Random Forest**
- Optimización de número de árboles, profundidad máxima y mínimo de muestras para división.
- `n_estimators = 200`.
- `max_depth = 5`.
- `min_samples_split = 5`.

Se utilizó GridSearchCV con validación cruzada Stratified K-Fold de cinco particiones y F1-score como criterio de optimización.

### 5. Resultados experimentales

| Métrica | Regresión logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.973684 | 0.956140 |
| F1-score | 0.979310 | 0.965517 |
| ROC-AUC | 0.995701 | 0.992725 |
| F1 promedio CV | 0.986116 | 0.970181 |
| Recall maligno | 0.952381 | 0.928571 |

### 6. Registro y trazabilidad mediante MLflow

Se configuró MLflow utilizando SQLite como backend de seguimiento de experimentos.

Los experimentos registraron:

- Algoritmo utilizado.
- Hiperparámetros seleccionados.
- Métricas de evaluación.
- Resultados de validación cruzada.
- Modelos entrenados.
- Identificadores únicos de ejecución (`run_id`).

Se generó un respaldo de los experimentos para conservar la evidencia del proyecto.

### 7. Análisis comparativo

La regresión logística presentó mejores resultados que Random Forest en las métricas evaluadas.

Su F1-score de prueba fue de 0.979310, frente a 0.965517 de Random Forest.

En la identificación de casos malignos, la regresión logística obtuvo un Recall de 0.952381, mientras que Random Forest alcanzó 0.928571.

Aunque ambos modelos presentaron resultados elevados, la regresión logística mostró un mejor equilibrio entre rendimiento y complejidad para este dataset.

### 8. Conclusiones

El proyecto permitió implementar un flujo reproducible de aprendizaje automático, desde la preparación y versionamiento de datos hasta el entrenamiento y registro de modelos.

Se comprobó la utilidad de GridSearchCV y la validación cruzada para comparar configuraciones de hiperparámetros.

MLflow permitió conservar los parámetros, métricas y modelos asociados a los experimentos.

**Se seleccionó regresión logística como el modelo con mejor desempeño**, de acuerdo con las métricas obtenidas.

Los resultados corresponden a una evaluación académica sobre un dataset público y no constituyen una validación para uso clínico.

### 9. Estado final

**Versión 1.0.0 — Implementación y ejecución completadas.**

Se dispone del código fuente, datasets versionados, notebook ejecutable y evidencia de experimentación con MLflow.
