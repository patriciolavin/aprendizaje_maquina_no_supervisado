# Proyecto 2: Segmentación de Preferencias Musicales a Nivel Global

**Tags:** `Machine Learning No Supervisado`, `Clustering`, `K-Means`, `Python`, `Análisis de Componentes Principales (PCA)`

## Objetivo del Proyecto

El objetivo de este proyecto es descubrir patrones y segmentar audiencias musicales a nivel global utilizando técnicas de **aprendizaje no supervisado**. A través de un dataset con características de canciones populares, se busca identificar grupos (clusters) de países con preferencias musicales similares, proveyendo insights valiosos para estrategias de marketing musical, recomendaciones de contenido y análisis de tendencias culturales.

## Metodología y Herramientas

El análisis se enfocó en agrupar datos no etiquetados para encontrar estructuras inherentes:

1.  **Análisis Exploratorio y Preprocesamiento:** Se utilizaron `Pandas` y `NumPy` para limpiar el dataset y escalar las características (`StandardScaler`), un paso crucial para algoritmos sensibles a la distancia como K-Means.
2.  **Reducción de Dimensionalidad (PCA):** Para visualizar los clusters en 2D y mitigar la "maldición de la dimensionalidad", se aplicó el **Análisis de Componentes Principales (PCA)**, reduciendo las múltiples características musicales a dos componentes principales que capturan la mayor parte de la varianza.
3.  **Modelado de Clustering:**
    * Se implementó el algoritmo **K-Means** de `Scikit-learn` para agrupar los países.
    * Se utilizó el **Método del Codo (Elbow Method)** para determinar el número óptimo de clusters (k) a generar.
4.  **Visualización e Interpretación:** Los clusters resultantes se visualizaron en un gráfico de dispersión usando `Matplotlib`, coloreando cada punto (país) según el segmento asignado.

## Resultados Clave

El algoritmo K-Means identificó exitosamente **[Número] clusters** distintos de preferencias musicales. Por ejemplo, se observó un cluster que agrupa a países con una fuerte preferencia por la música de alta energía y bailabilidad, mientras que otro cluster mostró una inclinación hacia la música acústica e instrumental. Estas segmentaciones permiten a la industria musical personalizar sus campañas a una escala regional precisa.

## Cómo Utilizar

1.  Clona este repositorio: `git clone https://github.com/patriciolavin/aprendizaje_maquina_no_supervisado.git`
2.  Instala las dependencias: `pip install pandas scikit-learn matplotlib seaborn`
3.  Ejecuta la Jupyter Notebook para replicar el análisis de clustering.
