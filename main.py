import logging
import yaml
from pathlib import Path
from src.data_handler import load_and_preprocess_data
from src.analysis import perform_full_analysis
from src.reporting import generate_report
import datetime # Importamos datetime para añadir la fecha al reporte

def main():
    """
    Orquestador principal del pipeline de análisis de preferencias musicales.
    Ejecuta la carga, preprocesamiento, análisis y generación de reportes.
    """
    # Construir la ruta al archivo de configuración de forma robusta
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / "config.yaml"

    # Cargar configuración desde la ruta correcta
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Configuración del logging centralizado
    log_path = SCRIPT_DIR / config['logging']['log_path']
    # Asegurarse de que el directorio de logs exista
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Pipeline de análisis iniciado.")

    try:
        # Construir rutas de datos y reportes basadas en SCRIPT_DIR
        raw_data_path = SCRIPT_DIR / config['data']['raw_path']
        bad_data_path = SCRIPT_DIR / config['data']['bad_data_path']
        template_folder = SCRIPT_DIR / config['reporting']['template_folder']
        report_path = SCRIPT_DIR / config['reporting']['report_path']
        report_path.parent.mkdir(parents=True, exist_ok=True) # Asegurar que la carpeta de reportes exista

        # --- Carga y Preprocesamiento de Datos ---
        logger.info("Paso 1: Cargando y preprocesando los datos.")
        preprocessor_pipeline, processed_data, country_names, bad_data_info = load_and_preprocess_data(
            raw_data_path,
            config['data']['country_column'],
            bad_data_path
        )
        logger.info(f"Preprocesamiento completado. Se guardó información de outliers en {bad_data_path}.")
        logger.info(f"Dimensiones de los datos procesados: {processed_data.shape}")

        # --- Análisis completo ---
        logger.info("Paso 2: Realizando el análisis de clustering y reducción de dimensionalidad.")
        analysis_results = perform_full_analysis(
            processed_data,
            country_names,
            config['analysis_params']
        )
        logger.info("Análisis completado exitosamente.")

        # --- Generación del Reporte ---
        logger.info("Paso 3: Generando el reporte de análisis en HTML.")
        
        # Diccionario 'analysis_params' y la fecha actual
        # para que la plantilla pueda acceder a ellos.
        report_context = {
            "date": datetime.date.today().strftime("%d-%m-%Y"),
            "params": config['analysis_params'],
            "bad_data_info": bad_data_info,
            "initial_shape": (len(country_names), processed_data.shape[1] + 1),
            **analysis_results
        }

        generate_report(
            report_context,
            template_folder,
            config['reporting']['template_name'],
            report_path
        )
        logger.info(f"Reporte generado y guardado en: {report_path}")

    except Exception as e:
        logger.critical(f"El pipeline ha fallado con un error crítico: {e}", exc_info=True)
    finally:
        logger.info("=" * 60)
        logger.info(" " * 2 + "✅ ACTIVIDAD FINAL DEL MÓDULO 7 COMPLETADA EXITOSAMENTE")
        logger.info("=" * 60 + "\n")

if __name__ == "__main__":
    main()