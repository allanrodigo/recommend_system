from pathlib import Path
import logging
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Configuração de diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

# Função para criar a matriz de características
def create_based_content_feature_matrix(df_products):
    """
    Cria a matriz de características baseada em atributos de produtos.

    Args:
        df_products (pd.DataFrame): DataFrame contendo os dados dos produtos.

    Returns:
        np.ndarray: Matriz de características.
        pd.Index: Índice dos produtos.
    """
    logging.info("Creating item feature matrix.")

    # Verificar se 'product_id' está presente
    if 'product_id' not in df_products.columns:
        raise ValueError("'product_id' column is required in the dataset.")

    df_products = df_products.set_index('product_id')
    # Preencher valores ausentes na coluna price com a média
    df_products['price'] = df_products['price'].fillna(df_products['price'].mean())

    # Seleção de colunas relevantes
    feature_columns = ['price']
    feature_columns += [col for col in df_products.columns if col.startswith('brand_')]
    feature_columns += [col for col in df_products.columns if col.startswith('category_')]

    # Preenchimento de valores nulos e construção da matriz de características
    based_content_feature_matrix = df_products[feature_columns].fillna(0).values
    logging.info(f"Feature matrix shape: {based_content_feature_matrix.shape}")

    return based_content_feature_matrix, df_products.index

# Função para salvar o modelo completo
def save_model(knn_model, feature_matrix, index, filename="model.pkl"):
    """
    Salva o modelo KNN, a matriz de características e o índice em um único arquivo.

    Args:
        knn_model (NearestNeighbors): Modelo KNN treinado.
        feature_matrix (np.ndarray): Matriz de características.
        index (pd.Index): Índice dos produtos.
        filename (str): Nome do arquivo para salvar o modelo.
    """
    filepath = MODELS_DIR / filename
    model = {
        "knn_model": knn_model,
        "feature_matrix": feature_matrix,
        "index": index,
    }
    joblib.dump(model, filepath)
    logging.info(f"Full model saved to {filepath}.")

# Função para carregar o modelo completo
def load_model(filename="model.pkl"):
    """
    Carrega o modelo KNN, a matriz de características e o índice de um único arquivo.

    Args:
        filename (str): Nome do arquivo que contém o modelo.

    Returns:
        dict: Um dicionário contendo o modelo, a matriz de características e o índice.
    """
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file {filename} not found in {MODELS_DIR}.")
    model = joblib.load(filepath)
    logging.info(f"Full model loaded from {filepath}.")
    return model

# Função para construir e salvar o modelo completo
def build_and_save_model(df_products):
    """
    Constrói e salva o modelo KNN, a matriz de características e o índice em um único arquivo.

    Args:
        df_products (pd.DataFrame): DataFrame contendo os dados dos produtos.
    """
    logging.info("Building full model.")

    # Criar matriz de características
    feature_matrix, index = create_based_content_feature_matrix(df_products)
    if not df_products['product_id'].is_unique:
        logging.warning("Duplicate product IDs found. Removing duplicates.")
        df_products = df_products.drop_duplicates(subset=['product_id'])
    index = df_products['product_id'].unique()

    # Construir o modelo KNN
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_model.fit(feature_matrix)

    # Salvar o modelo completo
    save_model(knn_model, feature_matrix, index)

# Função para gerar recomendações
def generate_recommendations(product_id: int, top_n=5, model_filename="model.pkl"):
    """
    Gera recomendações para um produto específico.

    Args:
        product_id (int or str): ID do produto para o qual as recomendações serão geradas.
        top_n (int): Número de recomendações (default: 10).
        model_filename (str): Nome do arquivo que contém o modelo.

    Returns:
        list: IDs dos produtos recomendados.
    """
    logging.info(f"Generating recommendations for product_id: {product_id}.")
    product_id = int(product_id)

    try:
        model = load_model(model_filename)
        knn_model = model["knn_model"]
        feature_matrix = model["feature_matrix"]
        index = model["index"]
        logging.info(f"Index type: {type(index)}")
        logging.info(f"First 5 values in index: {list(index[:5])}")
        logging.info(f"Product ID being searched: {product_id}")
        logging.info(f"Index dtype: {index.dtype}")
        logging.info(f"Product ID type: {type(product_id)}")
        logging.info(f"Feature matrix shape: {feature_matrix.shape}")
        logging.info(f"KNN model expects {knn_model.n_features_in_} features")

        if product_id not in index:
            raise ValueError(f"Product ID {product_id} not found in index.")

        item_idx_array = np.where(index == product_id)[0]
        if len(item_idx_array) != 1:
            logging.error("item_idx is not a single integer value. Debugging required.")
            raise ValueError(f"Invalid item_idx: {item_idx_array}")
        item_idx = item_idx_array[0]
        logging.info(f"Item index in feature matrix: {item_idx}")
        feature_vector = feature_matrix[item_idx].reshape(1, -1)
        logging.info(f"Feature vector shape: {feature_vector.shape}")

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
