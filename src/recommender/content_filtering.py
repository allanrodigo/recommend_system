from pathlib import Path
import pandas as pd
import logging
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# Configuração de diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

# Função para criar a matriz de características
def create_based_content_feature_matrix(df_products, n_components=100):
    """
    Cria a matriz de características baseada em atributos de produtos.

    Args:
        df_products (pd.DataFrame): DataFrame contendo os dados dos produtos.
        n_components (int): Número de componentes para redução de dimensionalidade.

    Returns:
        np.ndarray: Matriz reduzida de características.
        pd.Index: Índice dos produtos.
    """
    logging.info("Creating item feature matrix.")

    # Verificar se 'product_id' está presente
    if 'product_id' not in df_products.columns:
        raise ValueError("'product_id' column is required in the dataset.")

    df_products = df_products.set_index('product_id')

    # Seleção de colunas relevantes
    feature_columns = ['price']
    feature_columns += [col for col in df_products.columns if col.startswith('brand_')]
    feature_columns += [col for col in df_products.columns if col.startswith('category_')]

    # Preenchimento de valores nulos e construção da matriz de características
    based_content_feature_matrix = df_products[feature_columns].fillna(0)
    logging.info(f"Feature matrix shape before SVD: {based_content_feature_matrix.shape}")

    # Aplicar TruncatedSVD para redução de dimensionalidade
    logging.info(f"Applying dimensionality reduction with {n_components} components.")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    based_content_feature_matrix_reduced = svd.fit_transform(based_content_feature_matrix)
    logging.info(f"Feature matrix shape after SVD: {based_content_feature_matrix_reduced.shape}")

    # Salvar a matriz reduzida e o modelo SVD
    save_model(based_content_feature_matrix_reduced, "based_content_feature_matrix.pkl")
    save_model(svd, "based_content_svd_model.pkl")

    logging.info("Feature matrix and SVD model saved successfully.")
    return based_content_feature_matrix_reduced, df_products.index

# Função para salvar modelos
def save_model(obj, filename):
    """
    Salva um objeto serializado no diretório de modelos.

    Args:
        obj: Objeto a ser salvo.
        filename (str): Nome do arquivo.
    """
    filepath = MODELS_DIR / filename
    joblib.dump(obj, filepath)
    logging.info(f"Model saved to {filepath}.")

# Função para carregar modelos
def load_model(filename):
    """
    Carrega um modelo serializado do diretório de modelos.

    Args:
        filename (str): Nome do arquivo.

    Returns:
        obj: Modelo carregado.
    """
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file {filename} not found in {MODELS_DIR}.")
    return joblib.load(filepath)

# Função para construir e salvar o modelo KNN
def build_and_save_knn_model(based_content_feature_matrix, index, metric='cosine', algorithm='brute'):
    """
    Constrói e salva o modelo KNN.

    Args:
        based_content_feature_matrix (np.ndarray): Matriz de características reduzida.
        index (pd.Index): Índice dos produtos.
        metric (str): Métrica de similaridade (default: 'cosine').
        algorithm (str): Algoritmo para encontrar vizinhos (default: 'brute').

    Returns:
        NearestNeighbors: Modelo KNN treinado.
    """
    logging.info("Building NearestNeighbors model.")
    knn_model = NearestNeighbors(metric=metric, algorithm=algorithm)
    knn_model.fit(based_content_feature_matrix)

    # Salvar modelo e índice
    save_model(knn_model, "based_content_knn_model.pkl")
    save_model(index, "based_content_index.pkl")

    logging.info("KNN model and index saved successfully.")
    return knn_model

# Função para gerar recomendações
def generate_recommendations(product_id, top_n=10):
    """
    Gera recomendações para um produto específico.

    Args:
        product_id (int or str): ID do produto para o qual as recomendações serão geradas.
        top_n (int): Número de recomendações (default: 10).

    Returns:
        list: IDs dos produtos recomendados.
    """
    logging.info(f"Generating recommendations for product_id: {product_id}.")

    try:
        # Carregar modelos
        knn_model = load_model("based_content_knn_model.pkl")
        index = load_model("based_content_index.pkl")
        feature_matrix = load_model("based_content_feature_matrix.pkl")
        logging.info(f"Feature matrix shape: {feature_matrix.shape}")

        # Validar se o produto está no índice
        if product_id not in index:
            raise ValueError(f"Product ID {product_id} not found in index.")

        item_idx = index.get_loc(product_id)
        logging.info(f"Item index in feature matrix: {item_idx}")
        
        feature_vector = feature_matrix[item_idx]
        logging.info(f"Input vector shape after SVD: {feature_vector.shape}")

        # Buscar vizinhos mais próximos
        _, indices = knn_model.kneighbors(
            feature_vector,
            n_neighbors=top_n + 1
        )

        # Excluir o próprio produto dos resultados
        recommended_indices = indices.flatten()[1:top_n + 1]
        recommendations = index[recommended_indices].tolist()
        logging.info(f"Recommendations for product {product_id}: {recommendations}")
        
        return recommendations
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        raise

# Função principal para treinamento e geração de recomendações
def content_filtering(df: pd.DataFrame):
    """
    Realiza todo o processo de treinamento e geração de recomendações.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados dos produtos.
    """
    # Criar matriz de características
    based_content_feature_matrix, index = create_based_content_feature_matrix(df)

    # Construir e salvar o modelo KNN
    knn_model = build_and_save_knn_model(based_content_feature_matrix, index)

    # Imprimir métricas do modelo
    logging.info(f"Number of samples in the model: {knn_model.n_samples_fit_}")
