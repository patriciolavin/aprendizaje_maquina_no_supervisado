import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import euclidean_distances
import io
import base64
import logging

logger = logging.getLogger(__name__)

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convierte una figura de matplotlib a una cadena base64 para HTML.
    Cierra la figura para liberar memoria.
    """
    buf = io.BytesIO()
    # bbox_inches='tight' asegura que no se corten las etiquetas o títulos
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    # plt.close() es crucial para evitar que las figuras se acumulen en memoria durante la ejecución
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ==============================================================================
# FUNCIÓN DEDICADA AL ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================

def perform_eda(df: pd.DataFrame) -> dict:
    """
    Realiza un Análisis Exploratorio de Datos (EDA) completo sobre el dataframe limpio
    (antes de escalar) y devuelve estadísticas y visualizaciones.

    Args:
        df (pd.DataFrame): El DataFrame con los datos limpios y en su escala original.

    Returns:
        dict: Un diccionario que contiene todas las estadísticas y gráficos del EDA.
    """
    logger.info("Iniciando el Análisis Exploratorio de Datos (EDA) detallado.")
    eda_results = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --- 1. Información General y Estadísticas Descriptivas ---
    logger.info("Calculando estadísticas descriptivas, asimetría y curtosis.")
    # Usamos to_html() para renderizar fácilmente la tabla en el reporte
    eda_results['descriptive_stats_html'] = df.describe().to_html(classes='table table-striped table-hover table-sm')
    eda_results['skewness'] = df[numeric_cols].skew().to_dict()
    eda_results['kurtosis'] = df[numeric_cols].kurtosis().to_dict()

    # --- 2. Generación de Visualizaciones ---
    visualizations = {}

    # Matriz de Correlación
    logger.info("Generando matriz de correlación.")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Matriz de Correlación de Géneros Musicales', fontsize=16)
    visualizations['correlation_heatmap'] = fig_to_base64(fig)

    # Histogramas de Distribución
    logger.info("Generando histogramas para variables numéricas.")
    histograms = {}
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col], kde=True, ax=ax, bins=5) # Ajustar bins para dataset pequeño
        ax.set_title(f'Distribución de {col}', fontsize=14)
        histograms[col] = fig_to_base64(fig)
    visualizations['histograms'] = histograms

    # Box Plots para Análisis de Dispersión
    logger.info("Generando box plots para variables numéricas.")
    box_plots = {}
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Box Plot de {col}', fontsize=14)
        box_plots[col] = fig_to_base64(fig)
    visualizations['box_plots'] = box_plots
    
    eda_results['visualizations'] = visualizations
    logger.info("EDA detallado completado.")
    return eda_results

# ==============================================================================
# SECCIÓN DE CLUSTERING Y REDUCCIÓN DE DIMENSIONALIDAD
# ==============================================================================

def plot_clusters(data: np.ndarray, labels: np.ndarray, title: str, country_names: list, cluster_centers: np.ndarray = None) -> plt.Figure:
    """
    Genera un gráfico de clusters con nombres de países anotados.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(np.unique(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    
    for k, col in zip(unique_labels, palette):
        if k == -1: col = 'k'  # Ruido en DBSCAN se grafica en negro
        
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], label=f'Cluster {k}')

    for i, txt in enumerate(country_names):
        ax.annotate(txt, (data[i, 0], data[i, 1]), fontsize=9, ha='right')
        
    if cluster_centers is not None:
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroides')
        
    ax.set_title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    return fig

def perform_full_analysis(data: np.ndarray, country_names: list, params: dict) -> dict:
    """
    Ejecuta el pipeline completo de análisis sobre los datos escalados, incluyendo
    clustering y reducción de dimensionalidad.
    """
    random_state = params['random_state']
    n_samples = data.shape[0]
    analysis_results = {}
    
    # --- General Dataset Overview ---
    logger.info("Agregando primeras filas y estadísticas generales del dataset al reporte.")
    analysis_results['head_html'] = pd.DataFrame(data, columns=None if not hasattr(country_names, '__len__') else country_names).head().to_html(classes='table table-striped table-hover table-sm', index=False)
    analysis_results['shape'] = data.shape
    analysis_results['columns'] = country_names if hasattr(country_names, '__len__') else [f'col_{i}' for i in range(data.shape[1])]
    
    # Estadísticas generales del dataset
    analysis_results['general_stats'] = {
        'n_samples': n_samples,
        'n_features': data.shape[1],
        'feature_names': analysis_results['columns'],
        'min': np.min(data, axis=0).tolist(),
        'max': np.max(data, axis=0).tolist(),
        'mean': np.mean(data, axis=0).tolist(),
        'std': np.std(data, axis=0).tolist()
    }
    
    
    # --- K-Means Analysis ---
    logger.info("Ejecutando K-Means...")
    max_k_allowed = min(params['kmeans']['k_range'][-1], n_samples - 1)
    
    if max_k_allowed < 2:
        logger.error(f"No se puede ejecutar K-Means. Se necesitan al menos 2 clusters, pero el dataset solo permite hasta {max_k_allowed}.")
        return {} 

    k_range = range(2, max_k_allowed + 1)
    logger.info(f"Rango de K para K-Means ajustado a la cantidad de datos: {list(k_range)}")

    inertias, silhouette_scores = [], []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto').fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    fig_elbow = plt.figure()
    plt.plot(k_range, inertias, 'bo-')
    plt.title('Método del Codo para K-Means'); plt.xlabel('Número de Clusters (k)'); plt.ylabel('Inercia')
    analysis_results['kmeans_elbow_plot'] = fig_to_base64(fig_elbow)
    
    fig_sil = plt.figure()
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.title('Coeficiente de Silueta para K-Means'); plt.xlabel('Número de Clusters (k)'); plt.ylabel('Coeficiente de Silueta')
    analysis_results['kmeans_silhouette_plot'] = fig_to_base64(fig_sil)
    
    optimal_k = k_range[np.argmax(silhouette_scores)] if silhouette_scores else 2
    analysis_results['kmeans_optimal_k'] = optimal_k
    logger.info(f"K óptimo según silueta: {optimal_k}")
    
    k_initial = min(params['kmeans']['initial_k'], max_k_allowed)
    kmeans_initial = KMeans(n_clusters=k_initial, random_state=random_state, n_init='auto').fit(data)
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=random_state, n_init='auto').fit(data)
    analysis_results['kmeans_initial_labels'] = kmeans_initial.labels_
    analysis_results['kmeans_optimal_labels'] = kmeans_optimal.labels_
    analysis_results['kmeans_optimal_centroids'] = kmeans_optimal.cluster_centers_

    # --- Hierarchical Clustering ---
    logger.info("Ejecutando Clustering Jerárquico...")
    linked = linkage(data, method='ward')
    fig_dendrogram, ax = plt.subplots(figsize=(12, 7))
    dendrogram(linked, orientation='top', labels=country_names, distance_sort='descending', show_leaf_counts=True, ax=ax)
    ax.set_title('Dendrograma del Clustering Jerárquico'); ax.set_ylabel("Distancia de Ward")
    analysis_results['hierarchical_dendrogram'] = fig_to_base64(fig_dendrogram)
    
    hierarchical_cluster = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    analysis_results['hierarchical_labels'] = hierarchical_cluster.fit_predict(data)

    # --- DBSCAN Analysis ---
    logger.info("Ejecutando DBSCAN...")
    best_dbscan_score, best_dbscan_params, best_dbscan_labels = -1, {}, None
    for eps in params['dbscan']['eps_range']:
        for min_samples in params['dbscan']['min_samples_range']:
            if min_samples >= n_samples: continue
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
            if 1 < len(np.unique(labels)) < n_samples:
                score = silhouette_score(data, labels)
                if score > best_dbscan_score:
                    best_dbscan_score, best_dbscan_params, best_dbscan_labels = score, {'eps': eps, 'min_samples': min_samples}, labels
    
    if best_dbscan_labels is not None:
        analysis_results['dbscan_labels'], analysis_results['dbscan_params'] = best_dbscan_labels, best_dbscan_params
    else:
        logger.warning("No se encontró una configuración óptima para DBSCAN, usando valores por defecto.")
        dbscan_default = DBSCAN(eps=1.5, min_samples=2).fit(data)
        analysis_results['dbscan_labels'], analysis_results['dbscan_params'] = dbscan_default.labels_, {'eps': 2.5, 'min_samples': 2}

    # --- Dimensionality Reduction ---
    logger.info("Ejecutando PCA...")
    pca = PCA(random_state=random_state)
    pca_data = pca.fit_transform(data)
    exp_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(exp_var_ratio >= params['pca']['explained_variance_threshold']) + 1 if np.any(exp_var_ratio >= params['pca']['explained_variance_threshold']) else len(exp_var_ratio)

    fig_pca_var = plt.figure()
    plt.plot(range(1, len(exp_var_ratio) + 1), exp_var_ratio, 'o-'); plt.title('Varianza explicada por Componentes Principales')
    plt.axhline(y=params['pca']['explained_variance_threshold'], color='r', linestyle='--'); plt.axvline(x=n_components_90, color='g', linestyle='--')
    analysis_results['pca_variance_plot'] = fig_to_base64(fig_pca_var)
    analysis_results['pca_n_components_90'] = n_components_90
    
    # Proyectar centroides para visualización
    pca_centroids = pca.transform(kmeans_optimal.cluster_centers_)
    fig_pca_clusters = plot_clusters(pca_data[:, :2], analysis_results['kmeans_optimal_labels'], 'Clusters K-Means (visualizados con PCA)', country_names, pca_centroids[:,:2])
    analysis_results['pca_kmeans_plot'] = fig_to_base64(fig_pca_clusters)

    logger.info("Ejecutando t-SNE...")
    tsne_plots = {}
    max_perplexity = n_samples - 1
    for perplexity in params['tsne']['perplexity_range']:
        if perplexity > max_perplexity:
            logger.warning(f"Perplexity {perplexity} es demasiado alto para n_samples={n_samples}. Se omite.")
            continue
        tsne_data = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='pca', learning_rate='auto').fit_transform(data)
        fig_tsne = plot_clusters(tsne_data, analysis_results['kmeans_optimal_labels'], f'Clusters K-Means (t-SNE, Perplexity={perplexity})', country_names)
        tsne_plots[f'perplexity_{perplexity}'] = fig_to_base64(fig_tsne)
    analysis_results['tsne_plots'] = tsne_plots

    return analysis_results