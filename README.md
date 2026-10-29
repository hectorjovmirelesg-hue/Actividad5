# Actividad 5 — Entrenamiento y Evaluación de Modelos con MLflow

## 1. Descripción

Proyecto académico de Machine Learning que implementa la preparación y versionamiento de datos, entrenamiento de modelos, optimización de hiperparámetros, validación cruzada y seguimiento de experimentos mediante MLflow.

Se comparan dos algoritmos de clasificación supervisada: regresión logística y Random Forest.

## 2. Dataset

**Breast Cancer Wisconsin (Diagnostic)**

- Fuente: UCI Machine Learning Repository, disponible en Scikit-learn.
- Registros: 569.
- Variables predictoras: 30.
- Variable objetivo: `target`.
- Clases: 0 = maligno; 1 = benigno.

## 3. Estructura del repositorio

```text
Actividad5/
├── datos/
│   ├── datos_ini/
│   │   └── breast_cancer_original.csv
│   └── datos_limp/
│       ├── breast_cancer_limpio.csv
│       ├── resultados_iniciales.csv
│       └── resultados_optimizados.csv
├── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
├── README.md
└── CHANGELOG.md
```

## 4. Tecnologías

- Python 3
- Google Colab
- Pandas
- Scikit-learn
- MLflow
- SQLite
- Git y GitHub

## 5. Metodología

1. Carga del dataset público.
2. Validación de valores nulos, duplicados y tipos de datos.
3. Generación del dataset limpio.
4. División estratificada: 80% entrenamiento y 20% prueba.
5. Entrenamiento de regresión logística y Random Forest.
6. Optimización de hiperparámetros mediante GridSearchCV.
7. Validación cruzada Stratified K-Fold con cinco particiones.
8. Evaluación con Accuracy, Precision, Recall, F1-score y ROC-AUC.
9. Registro de parámetros, métricas y modelos en MLflow.
10. Comparación de resultados y generación de evidencias.

## 6. Reproducción del proyecto

### Requisitos

- Cuenta de Google para utilizar Google Colab.
- Acceso a Internet.
- Bibliotecas Python utilizadas en el notebook.

### Instrucciones

1. Abrir `fuentes/entrena.ipynb` desde Google Colab.
2. Revisar la configuración del repositorio GitHub en las celdas correspondientes.
3. Instalar las dependencias indicadas en el notebook.
4. Seleccionar **Entorno de ejecución → Ejecutar todo**.
5. Esperar a que finalicen la preparación, el entrenamiento y la optimización.
6. Verificar las métricas y los registros de MLflow.
7. Exportar los resultados y el respaldo de los experimentos.

El notebook integra la ejecución de `datos_prep.py` y `train.py`, disponibles en la carpeta `fuentes`.

**Nota:** el almacenamiento SQLite utilizado en Colab es temporal. Es necesario descargar el respaldo de MLflow para conservar los experimentos.

## 7. Resultados

| Métrica | Regresión logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.973684 | 0.956140 |
| F1-score | 0.979310 | 0.965517 |
| ROC-AUC | 0.995701 | 0.992725 |
| F1 promedio CV | 0.986116 | 0.970181 |
| Recall maligno | 0.952381 | 0.928571 |

## 8. Mejores hiperparámetros

**Regresión logística**

- `C = 0.1`
- `max_iter = 2000`

**Random Forest**

- `n_estimators = 200`
- `max_depth = 5`
- `min_samples_split = 5`

## 9. Conclusiones

La regresión logística obtuvo mejores resultados que Random Forest en las métricas evaluadas.

La validación cruzada y GridSearchCV permitieron comparar diferentes configuraciones de los modelos.

MLflow facilitó la trazabilidad de los experimentos al registrar parámetros, métricas y modelos entrenados.

Los resultados son académicos y no representan una validación para aplicaciones clínicas.

## 10. Registro de cambios

Consultar `CHANGELOG.md` para revisar las decisiones técnicas, los resultados experimentales y las conclusiones del proyecto.

## 11. Estado

**Versión 1.0.0 — Proyecto completado y ejecutado correctamente en Google Colab.**
