
# Actividad 5
# Preparación y limpieza de datos

import pandas as pd

def preparar_datos(ruta_entrada, ruta_salida):
    # Cargar datos originales
    df = pd.read_csv(ruta_entrada)

    print("Dimensiones originales:", df.shape)

    # Validar valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # Validar registros duplicados
    print("\nRegistros duplicados:", df.duplicated().sum())

    # Validar tipos de datos
    print("\nTipos de datos:")
    print(df.dtypes)

    # Eliminar duplicados
    df = df.drop_duplicates().copy()

    # Verificar tipos de las variables predictoras
    columnas = df.columns.drop("target")
    for columna in columnas:
        df[columna] = pd.to_numeric(df[columna], errors="raise")

    # Validar variable objetivo
    df["target"] = pd.to_numeric(df["target"], errors="raise")

    if df["target"].isna().any():
        raise ValueError("La variable objetivo contiene valores nulos")

    if not df["target"].isin([0, 1]).all():
        raise ValueError("La variable objetivo contiene clases inválidas")

    # Validar que no existan nulos en las variables predictoras
    if df[columnas].isna().any().any():
        raise ValueError("Existen valores nulos que requieren tratamiento")

    # Guardar dataset limpio
    df.to_csv(ruta_salida, index=False)

    print("\nDimensiones finales:", df.shape)
    print("Dataset limpio guardado correctamente")

    return df
