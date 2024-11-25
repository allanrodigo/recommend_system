
from pathlib import Path
from config import setup_logging
import logging
import pandas as pd

from src.recommender.content_filtering import build_and_save_model, generate_recommendations


setup_logging()

def main():
    logging.info("Start application")

    # Caminho do arquivo de entrada
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "data" / "processed" / "processed_data_pro.csv"

    # Criar DataFrame
    logging.info("Create dataframe from processed data")
    df = pd.read_csv(input_path, low_memory=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    columns_to_replace = [col for col in df.columns if col.startswith(("event_type", "brand", "category"))]
    df[columns_to_replace] = df[columns_to_replace].replace({"False": 0, "True": 1})
    logging.info("DataFrame created successfully")

    # Construir e salvar o modelo completo
    build_and_save_model(df)
    generate_recommendations(29900078)
if __name__ == "__main__":
    main()