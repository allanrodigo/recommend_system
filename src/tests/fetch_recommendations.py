import pandas as pd
import requests
import json
import logging
from tqdm import tqdm

# Configurações da API
API_URL = "http://127.0.0.1:8000/recommend"
INPUT_CSV = "C:\\Users\\allan\\Documents\\recommendation_system\\data\\raw\\dataset-sales-minor-pro.csv"
OUTPUT_FILE = "recommendations_10k.json"
TOP_N = 5  # Número de recomendações por produto

def fetch_recommendations(product_ids, top_n=TOP_N):
    """
    Faz requests para a API de recomendação para os product_ids fornecidos.
    
    Args:
        product_ids (list): Lista de IDs de produtos.
        top_n (int): Número de recomendações a serem retornadas.

    Returns:
        dict: Um dicionário com os product_ids e suas recomendações.
    """
    recommendations = {}

    for product_id in tqdm(product_ids, desc="Fetching recommendations"):
        try:
            response = requests.get(API_URL, params={"product_id": product_id, "top_n": top_n})

            # Verificar se a resposta é válida
            if response.status_code == 200:
                data = response.json()
                recommendations[product_id] = data["recommendations"]
            else:
                logging.warning(f"Request failed for product_id {product_id}. Status code: {response.status_code}")
                recommendations[product_id] = None  # Marcar como falha
        except Exception as e:
            logging.error(f"Erro ao fazer request para product_id {product_id}: {e}")
            recommendations[product_id] = None  # Marcar como falha
    
    return recommendations


def main():
    # Ler IDs de produtos do arquivo CSV
    try:
        df = pd.read_csv(INPUT_CSV)
        if "product_id" not in df.columns:
            raise ValueError("O arquivo CSV deve conter uma coluna chamada 'product_id'.")
        
        product_ids = df["product_id"].drop_duplicates().tolist()[:1000]
        logging.info(f"Carregados {len(product_ids)} product_ids do arquivo {INPUT_CSV}.")
    except Exception as e:
        logging.error(f"Erro ao carregar o arquivo CSV: {e}")
        return

    # Fazer requests para a API
    logging.info("Iniciando requests para a API de recomendação.")
    recommendations = fetch_recommendations(product_ids)

    # Salvar resultados em arquivo JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(recommendations, f, indent=4)
    
    logging.info(f"Recomendações salvas em {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
