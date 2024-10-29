from pathlib import Path
import joblib
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
import logging

def collaborate_filtering(df: pd.DataFrame):
    logging.info("Starting collaborative filtering model training.")
    event_weight_map = {
    'event_type_view': 1,
    'event_type_cart': 2,
    'event_type_purchase': 3
    }
    
        # Ensure 'event_type_*' columns are present
    logging.info("Ensuring 'event_type_*' columns are present.")
    for col in event_weight_map.keys():
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in DataFrame. Adding column with zeros.")
            df[col] = 0

    # Convert 'event_type_*' columns to booleans
    logging.info("Converting 'event_type_*' columns to booleans.")
    for col in event_weight_map.keys():
        df[col] = df[col].map({'False': False, 'True': True})
        df[col] = df[col].astype(bool)
        logging.debug(f"Unique values in '{col}': {df[col].unique()}")

    # Convert booleans to integers
    logging.info("Converting 'event_type_*' columns to integers.")
    for col in event_weight_map.keys():
        df[col] = df[col].astype(int)
        logging.debug(f"Unique values in '{col}': {df[col].unique()}")
    
    logging.info("Calculating event weights based on event type columns.")
    df['event_weight'] = (
        df['event_type_view'] * event_weight_map['event_type_view'] +
        df['event_type_cart'] * event_weight_map['event_type_cart'] +
        df['event_type_purchase'] * event_weight_map['event_type_purchase']
    )
    logging.info(f"Event weights calculated. DataFrame shape: {df.shape}")
    
    logging.info("Converting 'user_id' and 'product_id' to string type.")
    df['user_id'] = df['user_id'].astype(str)
    df['product_id'] = df['product_id'].astype(str)
    
    df.dropna(subset=['user_id', 'product_id', 'event_weight'], inplace=True)
    
    # Aggregate multiple interactions (keep the highest weight)
    df = df.groupby(['user_id', 'product_id'])['event_weight'].max().reset_index()
    
    logging.info("Defining the rating scale for the Surprise Reader.")
    reader = Reader(rating_scale=(1, 3))
    
    # Load data into Surprise dataset
    logging.info("Loading data into Surprise dataset.")
    data = Dataset.load_from_df(df[['user_id', 'product_id', 'event_weight']], reader)
    
       # Build the full trainset
    logging.info("Building the full trainset for GridSearchCV.")
    full_trainset = data.build_full_trainset()

    # Hyperparameter tuning using GridSearchCV
    logging.info("Starting hyperparameter tuning with GridSearchCV.")
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30, 40],
        'lr_all': [0.002, 0.005, 0.008],
        'reg_all': [0.02, 0.05, 0.1]
    }

    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
    gs.fit(data)

    logging.info(f"Best RMSE score from GridSearchCV: {gs.best_score['rmse']:.4f}")
    logging.info(f"Best hyperparameters: {gs.best_params['rmse']}")

    # Train the SVD model with the best found hyperparameters on the full trainset
    logging.info("Training the final SVD model with the best hyperparameters.")
    best_model = gs.best_estimator['rmse']
    best_model.fit(full_trainset)
    logging.info("Final model training completed.")

    # Evaluate the model on a test set
    logging.info("Evaluating the final model performance.")
    
    logging.info("Saving the trained model.")
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_path = base_dir / 'models' / 'collaborate_recommendation_model.pkl'
    logging.info(f"Saving the model to {model_path}.")
    joblib.dump(best_model, model_path)
    logging.info("Model trained and saved successfully!")
