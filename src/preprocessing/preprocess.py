import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Configurações globais
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_FILE = BASE_DIR / 'data' / 'processed' / 'processed_data_pro.csv'

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando o pré-processamento do conjunto de dados.")

    # Preencher valores ausentes em 'category_code' e 'brand' com 'unknown'
    df['category_code'] = df['category_code'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')

    # Separar colunas de categoria
    categories = df['category_code'].str.split('.', n=2, expand=True)
    df['category_main'] = categories[0].fillna('unknown')
    df['category_sub'] = categories[1].fillna('unknown')
    df['category_sub_2'] = categories[2].fillna('unknown')

    # Converter 'price' para numérico
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Agrupar por 'product_id' e calcular o preço médio
    df_price = df.groupby('product_id')['price'].mean().reset_index()

    # Preencher valores ausentes em 'price' com a média geral (excluindo NaN)
    overall_mean_price = df_price['price'].mean(skipna=True)
    df_price['price'] = df_price['price'].fillna(overall_mean_price)

    # Normalizar o preço
    scaler = MinMaxScaler()
    df_price['n_price'] = scaler.fit_transform(df_price[['price']])

    # Contar interações por 'product_id' e 'event_type'
    interaction_counts = df.pivot_table(
        index='product_id',
        columns='event_type',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Obter 'brand' e categorias mais frequentes por 'product_id'
    df_categorical = df.groupby('product_id').agg({
        'brand': lambda x: x.mode(dropna=False).iloc[0] if not x.mode(dropna=False).empty else 'unknown',
        'category_main': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        'category_sub': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        'category_sub_2': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
    }).reset_index()

    # Converter colunas categóricas para string
    categorical_features = ['brand', 'category_main', 'category_sub', 'category_sub_2']
    df_categorical[categorical_features] = df_categorical[categorical_features].astype(str)

    # Aplicar OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cats = encoder.fit_transform(df_categorical[categorical_features])

    # Obter nomes das novas features
    encoded_cat_names = encoder.get_feature_names_out(categorical_features)

    # Criar DataFrame com as novas features
    df_encoded = pd.DataFrame(
        encoded_cats,
        columns=encoded_cat_names,
        index=df_categorical['product_id']
    )

    # Combinar todas as features
    feature_matrix = df_price.set_index('product_id').join(
        interaction_counts.set_index('product_id'), how='left'
    ).join(
        df_encoded, how='left'
    ).fillna(0)

    # Resetar o índice para ter 'product_id' como coluna
    feature_matrix = feature_matrix.reset_index()

    logging.info(f"Pré-processamento concluído. Dimensão da matriz de features: {feature_matrix.shape}")

    return feature_matrix

# Função para carregar e processar os dados
def load_and_preprocess_data(input_file: str) -> pd.DataFrame:
    logging.info("Loading raw data.")
    input_path = BASE_DIR / 'data' / 'raw' / input_file
    df = pd.read_csv(input_path, low_memory=False)
    logging.info(f"Raw data loaded. Shape: {df.shape}")

    processed_data = preprocess_data(df)
    logging.info("Processed data saved successfully.")
    return processed_data

if __name__ == "__main__":
    input_file = "C:\\Users\\allan\\Documents\\recommendation_system\\data\\raw\\dataset-sales-minor-pro.csv"
    processed_data = load_and_preprocess_data(input_file)

    # Salvar dados processados
    processed_data.to_csv(PROCESSED_FILE, index=False)
    logging.info(f"Processed data saved to {PROCESSED_FILE}")
