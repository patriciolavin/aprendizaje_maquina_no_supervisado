import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
import os
from io import StringIO


logger = logging.getLogger(__name__)

def detect_and_handle_outliers(df: pd.DataFrame, bad_data_path: str):
    """
    Detecta outliers utilizando el método del rango intercuartílico (IQR).
    Guarda los outliers (incluyendo columnas no numéricas como 'País') en un archivo CSV con metadatos.
    """
    numeric_df = df.select_dtypes(include='number')
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detectamos qué filas tienen al menos un outlier
    outlier_condition = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)
    outliers_df = df[outlier_condition]  # incluye columnas no numéricas

    outlier_count = len(outliers_df)
    percentage = (outlier_count / len(df)) * 100
    bad_data_info = {
        "outlier_count": outlier_count,
        "percentage": f"{percentage:.2f}%",
        "detection_method": "IQR (1.5 * Rango Intercuartílico)"
    }

    if outlier_count > 0:
        logger.warning(f"Se detectaron {outlier_count} outliers ({percentage:.2f}%). Exportando a {bad_data_path}")
        
        os.makedirs(os.path.dirname(bad_data_path), exist_ok=True)

        # Crear metadatos como comentarios
        metadata = "\n".join([
            "# Detalles de Outliers Detectados",
            *[f"# {key}: {value}" for key, value in bad_data_info.items()],
            "# ---"
        ])

        # Convertir DataFrame a CSV en memoria
    if outlier_count > 0:
        logger.warning(f"Se detectaron {outlier_count} outliers ({percentage:.2f}%). Exportando a {bad_data_path}")
        
        os.makedirs(os.path.dirname(bad_data_path), exist_ok=True)

        with open(bad_data_path, 'w', encoding='utf-8') as f:
            f.write(f"# Detalles de Outliers Detectados\n")
            for key, value in bad_data_info.items():
                f.write(f"# {key}: {value}\n")
            f.write("# ---\n")
    
    # Guardar con País como columna explícita
    outliers_df.reset_index().to_csv(bad_data_path, mode='a', index=False)
    logger.info("No se detectaron outliers significativos.")

    return df, bad_data_info


def load_and_preprocess_data(raw_path, country_column, bad_data_path):
    """
    Carga los datos, separa las características y los nombres, y crea un
    pipeline de preprocesamiento para escalar los datos.

    Retorna:
        - El pipeline de preprocesamiento ajustado.
        - Los datos procesados y escalados.
        - La lista de nombres de países.
        - Información sobre los datos anómalos (outliers).
    """
    try:
        df = pd.read_csv(raw_path)
        logger.info(f"Datos cargados desde {raw_path}. Dimensiones: {df.shape}")
    except FileNotFoundError:
        logger.error(f"El archivo de datos no se encontró en la ruta: {raw_path}")
        raise

    # Extraer nombres de países y características
    if country_column not in df.columns:
        logger.error(f"La columna '{country_column}' no se encuentra en el dataset.")
        raise ValueError(f"Missing column: {country_column}")
    
    # ⚠️ Detectar outliers antes de eliminar "País"
    df_with_outliers, bad_data_info = detect_and_handle_outliers(df, bad_data_path)
    
    # ⚠️ Luego separar nombre de países y características
    country_names = df_with_outliers[country_column].tolist()
    features = df_with_outliers.drop(columns=[country_column])

    # Manejo de Nulos (si los hubiera)
    if features.isnull().sum().sum() > 0:
        logger.warning("Valores nulos detectados. Se procederá a imputar con la media.")
        features = features.fillna(features.mean())
    
    # Aquí se podría añadir una estrategia de imputación más compleja si fuera necesario
    features = features.fillna(features.mean())
    
    # Detección de outliers (sin eliminarlos)
    df, bad_data_info = detect_and_handle_outliers(df, bad_data_path)

    # Creación del Pipeline de Preprocesamiento
    # Este pipeline encapsula el escalado, previniendo data leakage.
    # Si hubiera más pasos (imputación, encoding), se añadirían aquí.
    preprocessor_pipeline = Pipeline([
        # StandardScaler es sensible a outliers. Si la estrategia fuera eliminarlos, se usaría.
        # Dado que queremos un sistema robusto, podríamos usar RobustScaler,
        # pero para seguir el ejemplo clásico usaremos StandardScaler.
        ('scaler', StandardScaler())
    ])

    # Ajustar y transformar los datos con el pipeline
    processed_data = preprocessor_pipeline.fit_transform(features)
    
    # Obtener la carpeta root, asumiendo que este script está dentro de root/src/
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Subir 1 nivel a root
    
    # Obtener nombre base sin extensión
    basename = os.path.basename(raw_path)  # 'dataset_generos_musicales.csv'
    name_without_ext = os.path.splitext(basename)[0]  # 'dataset_generos_musicales'

    # Construir ruta absoluta a root/data/processed
    processed_dir = os.path.join(root_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    basename = os.path.basename(raw_path)
    name_without_ext = os.path.splitext(basename)[0]
    processed_path = os.path.join(processed_dir, f"{name_without_ext}_processed.csv")

    # Guardar los datos procesados (opcional)
    pd.DataFrame(processed_data, columns=features.columns).to_csv(processed_path, index=False)
    
    return preprocessor_pipeline, processed_data, country_names, bad_data_info