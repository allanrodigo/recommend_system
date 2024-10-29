from src.preprocessing.preprocess import preprocess_data_parallel
from src.recommender.collaborate_filtering import collaborate_filtering
from src.recommender.content_filtering import content_filtering
from src.recommender.hybrid_filtering import HybridRecommender
from pathlib import Path
from config import setup_logging
import logging
from pandas import read_csv, to_numeric


setup_logging()

def main():
    logging.info("Start application")

    df = preprocess_data_parallel()
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / 'data' / 'processed' / 'processed_data.csv'
    logging.info("Create dataframe from processed data")
    df = read_csv(input_path, low_memory=False)
    df['price'] = to_numeric(df['price'], errors='coerce')
    df = df.replace({'False': 0, 'True': 1})
    logging.info("Created successfully")
    collaborate_filtering(df)
    content_filtering(df)
    hybrid_recommender = HybridRecommender.load_models()
    hybrid_recommender.save_model()

if __name__ == "__main__":
    main()