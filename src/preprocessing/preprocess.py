from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
from src.utils.load_data import load_data
import logging

# Função de pré-processamento de um chunk
def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Processing chunk with {len(chunk)} rows.")
    
    scaler = MinMaxScaler()

    # Separar a coluna category_code em subcategorias
    chunk[['category_main', 'category_sub', 'category_sub_2']] = chunk['category_code'].str.split('.', n=2, expand=True)
    
    # One-Hot Encoding com sparse=True
    chunk = pd.get_dummies(chunk, columns=['event_type', 'brand', 'category_main', 'category_sub', 'category_sub_2'], sparse=True)
    
    # Normalizar a coluna price
    chunk['n_price'] = scaler.fit_transform(chunk[['price']])
    
    # Remover colunas desnecessárias
    chunk.drop(columns=['event_time', 'category_code', 'user_session'], inplace=True)
    
    logging.info("Chunk Finished.")
    return chunk

# Função para pré-processamento paralelo e salvamento incremental
def preprocess_data_parallel() -> None:
    logging.info("Starting data preprocessing.")
    
    # Carregar os dados
    df = load_data()
    logging.info(f"Data loaded successfully. Total rows: {len(df)}")
    
    df_reduced = df.sample(frac=0.70, random_state=42)
    
    # Definir número de chunks
    num_chunks = 50
    chunk_size = len(df_reduced) // num_chunks
    chunks = [df_reduced.iloc[i:i + chunk_size] for i in range(0, len(df_reduced), chunk_size)]

    base_dir = Path(__file__).resolve().parent.parent.parent
    output_file =  base_dir / 'data' / 'processed' / 'processed_data.csv'
    
    header_written = False
    try:
        logging.info("Attempting to process data in parallel using ProcessPoolExecutor.")
        
        # Usar ProcessPoolExecutor para paralelizar o processamento de chunks
        with ProcessPoolExecutor(max_workers=5) as executor:
            for chunk_result in executor.map(preprocess_chunk, chunks):
                # Salvar o chunk processado diretamente no arquivo CSV
                chunk_result.to_csv(output_file, mode='a', header=not header_written, index=False)
                header_written = True
    
    except OSError as e:
        logging.error(f"Parallel processing failed due to system resource limits: {e}")
        logging.info("Falling back to sequential processing.")
        
        # Processamento sequencial como fallback
        for chunk in chunks:
            chunk_result = preprocess_chunk(chunk)
            chunk_result.to_csv(output_file, mode='a', header=not header_written, index=False)
            header_written = True

    logging.info("Data preprocessing completed successfully.")

