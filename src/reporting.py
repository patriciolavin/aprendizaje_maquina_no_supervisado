from jinja2 import Environment, FileSystemLoader
import logging

logger = logging.getLogger(__name__)

def generate_report(context, template_folder, template_name, output_path):
    """
    Renderiza un diccionario de resultados en una plantilla Jinja2 y guarda el
    archivo HTML resultante.
    """
    try:
        env = Environment(loader=FileSystemLoader(template_folder))
        template = env.get_template(template_name)
        
        # Renderizar la plantilla con el contexto
        html_content = template.render(context)
        
        # Guardar el reporte
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Reporte HTML generado correctamente en {output_path}")

    except Exception as e:
        logger.error(f"Error al generar el reporte HTML: {e}")
        raise