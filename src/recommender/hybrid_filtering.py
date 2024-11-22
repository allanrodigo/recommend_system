from pathlib import Path
import joblib
import logging
import pandas as pd
import numpy as np

class HybridRecommender:
    def __init__(self, collaborative_similarity_df, knn_model, index, based_content_feature_matrix):
        self.collaborative_similarity_df = collaborative_similarity_df
        self.knn_model = knn_model
        self.index = index
        self.based_content_feature_matrix = based_content_feature_matrix

    @classmethod
    def load_models(cls):
        base_dir = Path(__file__).resolve().parent.parent.parent
        collaborative_similarity_path = base_dir / 'src' / 'models' / 'collaborate_recommendation_model.pkl'
        knn_model_path = base_dir / 'src' / 'models' / 'based_content_knn_model.pkl'
        index_path = base_dir / 'src' / 'models' / 'based_content_index.pkl'
        feature_matrix_path = base_dir / 'src' / 'models' / 'based_content_feature_matrix.pkl'

        try:
            # Load the precomputed collaborative similarity matrix
            collaborative_similarity_df = joblib.load(collaborative_similarity_path)
            logging.info("Collaborative model loaded successfully.")

            # Load the content-based KNN model and related data
            knn_model = joblib.load(knn_model_path)
            index = joblib.load(index_path)
            based_content_feature_matrix = joblib.load(feature_matrix_path)
            logging.info("Content-based KNN model and related data loaded successfully.")

            return cls(collaborative_similarity_df, knn_model, index, based_content_feature_matrix)
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading models: {e}")
            raise

    def recommend(self, product_id, top_n=10, alpha=0.5):
        """
        Generate hybrid recommendations for a given product_id.

        Parameters:
        - product_id: The ID of the product to base recommendations on.
        - top_n: Number of recommendations to generate.
        - alpha: Weighting factor to balance content-based and collaborative scores (0 <= alpha <= 1).

        Returns:
        - A list of recommended product IDs.
        """
        logging.info(f"Generating hybrid recommendations for product {product_id}.")

        # Check if the product exists in the index
        try:
            item_idx = self.index.get_loc(product_id)
        except KeyError:
            logging.warning(f"Product ID {product_id} not found in index.")
            return []

        # Content-based recommendations using KNN
        distances, indices = self.knn_model.kneighbors(
            self.based_content_feature_matrix[item_idx].reshape(1, -1),
            n_neighbors=top_n + 1  # +1 to include the product itself
        )
        # Flatten arrays
        distances = distances.flatten()
        indices = indices.flatten()

        # Exclude the product itself
        mask = indices != item_idx
        similar_indices = indices[mask][:top_n]
        content_scores = 1 - distances[mask][:top_n]  # Convert distances to similarity scores
        content_recommendations = self.index[similar_indices]

        # Collaborative recommendations
        if product_id not in self.collaborative_similarity_df.index:
            logging.warning(f"Product ID {product_id} not found in collaborative model.")
            # Use only content-based recommendations
            hybrid_recommendations = content_recommendations.tolist()
        else:
            # Get collaborative scores for the product
            collab_scores_series = self.collaborative_similarity_df[product_id].drop(product_id, errors='ignore')

            # Align collaborative scores with content-based recommendations
            collab_scores = collab_scores_series.reindex(content_recommendations).fillna(0).values

            # Normalize scores
            content_scores_norm = content_scores / np.linalg.norm(content_scores) if np.linalg.norm(content_scores) != 0 else content_scores
            collab_scores_norm = collab_scores / np.linalg.norm(collab_scores) if np.linalg.norm(collab_scores) != 0 else collab_scores

            # Combine scores
            hybrid_scores = alpha * content_scores_norm + (1 - alpha) * collab_scores_norm

            # Create DataFrame to sort the results
            recommendations_df = pd.DataFrame({
                'product_id': content_recommendations,
                'hybrid_score': hybrid_scores
            })
            recommendations_df = recommendations_df.sort_values(by='hybrid_score', ascending=False)

            # Get top N recommendations
            hybrid_recommendations = recommendations_df['product_id'].head(top_n).tolist()

        logging.info(f"Hybrid recommendations for product {product_id}: {hybrid_recommendations}")
        return hybrid_recommendations

    def save_model(self):
        base_dir = Path(__file__).resolve().parent.parent
        filepath = base_dir / 'models' / 'hybrid_recommender_model.pkl'
        try:
            joblib.dump(self, filepath)
            logging.info(f"HybridRecommender model saved successfully at {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")

    @classmethod
    def load_model(cls):
        filepath = Path("./src/models/hybrid_recommender_model.pkl")
        logging.info(f"Procurando pelo modelo no caminho: {filepath}")
        try:
            return joblib.load(filepath)
        except FileNotFoundError:
            logging.error(f"Model file not found at {filepath}")
            raise
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise