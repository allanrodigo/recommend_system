from pathlib import Path
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
from src.utils.load_data import load_data
import logging

# Configurações globais
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_FILE = BASE_DIR / 'data' / 'processed' / 'processed_data_pro.csv'

# Função de pré-processamento de um chunk
def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info(f"Processing chunk with {len(chunk)} rows.")
        
        # Normalizador para colunas numéricas
        scaler = MinMaxScaler()

        # Separar colunas de categoria
        chunk[['category_main', 'category_sub', 'category_sub_2']] = chunk['category_code'].str.split('.', n=2, expand=True)
        
        # One-Hot Encoding (convertendo para formato esparso para economia de memória)
        chunk = pd.get_dummies(chunk, columns=['event_type', 'brand', 'category_main', 'category_sub', 'category_sub_2'], sparse=True)
        
        # Normalizar colunas numéricas
        if 'price' in chunk.columns:
            chunk['n_price'] = scaler.fit_transform(chunk[['price']])
        else:
            chunk['n_price'] = np.nan

        # Remover colunas desnecessárias
        columns_to_drop = ['event_time', 'category_code', 'user_session']
        chunk.drop(columns=[col for col in columns_to_drop if col in chunk.columns], inplace=True, errors='ignore')
        
        logging.info(f"Chunk processing complete. Processed {len(chunk)} rows.")
        return chunk
    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        raise

# Função para salvar chunks processados incrementalmente
def save_processed_chunk(chunk: pd.DataFrame, header_written: bool):
    try:
        chunk.to_csv(PROCESSED_FILE, mode='a', header=not header_written, index=False)
        logging.info(f"Chunk saved successfully. {len(chunk)} rows written.")
    except Exception as e:
        logging.error(f"Error saving chunk: {e}")
        raise

# Função para pré-processamento paralelo e salvamento incremental
def preprocess_data_parallel(num_chunks=50, max_workers=5, sample_frac=0.7):
    try:
        logging.info("Starting data preprocessing.")
        
        # Carregar os dados
        df = load_data()
        logging.info(f"Data loaded successfully. Total rows: {len(df)}")

        # Reduzir o tamanho do dataset para amostragem, se necessário
        df_reduced = df.sample(frac=sample_frac, random_state=42)

        # Dividir o dataset em chunks
        chunk_size = max(len(df_reduced) // num_chunks, 1)
        chunks = [df_reduced.iloc[i:i + chunk_size] for i in range(0, len(df_reduced), chunk_size)]
        logging.info(f"Dataset split into {len(chunks)} chunks of size ~{chunk_size} rows each.")

        # Inicializar flag para cabeçalhos no arquivo
        header_written = False

        # Processar os chunks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chunk_result in executor.map(preprocess_chunk, chunks):
                save_processed_chunk(chunk_result, header_written)
                header_written = True

        logging.info("Data preprocessing completed successfully.")
    except OSError as e:
        logging.error(f"Parallel processing failed due to system resource limits: {e}")
        logging.info("Falling back to sequential processing.")
        
        # Processamento sequencial como fallback
        header_written = False
        for chunk in chunks:
            chunk_result = preprocess_chunk(chunk)
            save_processed_chunk(chunk_result, header_written)
            header_written = True
    except Exception as e:
        logging.error(f"Unexpected error in preprocessing pipeline: {e}")
        raise
    finally:
        gc.collect()
