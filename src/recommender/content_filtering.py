from pathlib import Path
import pandas as pd
import logging
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

def create_based_content_feature_matrix(df_products):
    logging.info("Creating item feature matrix.")
    
    # Ensure 'product_id' is set as the index
    df_products = df_products.set_index('product_id')

    # Select the feature columns
    feature_columns = ['price']

    # Identify brand and category columns
    brand_columns = [col for col in df_products.columns if col.startswith('brand_')]
    category_columns = [col for col in df_products.columns if col.startswith('category_')]

    feature_columns.extend(brand_columns)
    feature_columns.extend(category_columns)

    based_content_feature_matrix = df_products[feature_columns].fillna(0)

    # Apply dimensionality reduction to prevent memory issues
    svd = TruncatedSVD(n_components=100)
    based_content_feature_matrix_reduced = svd.fit_transform(based_content_feature_matrix)

    logging.info(f"Item feature matrix created with shape: {based_content_feature_matrix_reduced.shape}")
    base_dir = Path(__file__).resolve().parent.parent.parent
    feature_matrix_path = base_dir / 'models' / 'based_content_feature_matrix.pkl'
    logging.info(f"Saving the reduced feature matrix to {feature_matrix_path}.")
    joblib.dump(based_content_feature_matrix_reduced, feature_matrix_path)
    logging.info("Reduced feature matrix saved successfully!")
    return based_content_feature_matrix_reduced, df_products.index

def build_and_save_knn_model(based_content_feature_matrix, index):
    logging.info("Building NearestNeighbors model.")

    # Create and fit the NearestNeighbors model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(based_content_feature_matrix)

    # Save the model and index
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_path = base_dir / 'models' / 'based_content_knn_model.pkl'
    index_path = base_dir / 'models' / 'based_content_index.pkl'

    joblib.dump(knn_model, model_path)
    joblib.dump(index, index_path)

    logging.info("Model trained and saved successfully!")
    return knn_model

def print_model_metrics(knn_model):
    logging.info("Calculating model metrics.")
    # Print the number of samples in the model
    logging.info(f"Number of samples in the model: {knn_model.n_samples_fit_}")

def content_filtering(df: pd.DataFrame):
    # Create the item feature matrix
    based_content_feature_matrix, index = create_based_content_feature_matrix(df)

    # Build and save the KNN model
    knn_model = build_and_save_knn_model(based_content_feature_matrix, index)

    # Print model metrics
    print_model_metrics(knn_model)
