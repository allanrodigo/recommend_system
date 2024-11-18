from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required
from flask_cors import CORS
from pathlib import Path
import logging
import joblib

from config import API_KEY

app = Flask(__name__)
CORS(app)

# JWT Config
app.config['API-KEY'] = API_KEY
jwt = JWTManager(app)

# Carregar o modelo baseado em conteúdo e seus dados auxiliares
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'

try:
    content_knn_model = joblib.load(MODELS_DIR / 'based_content_knn_model.pkl')
    content_feature_matrix = joblib.load(MODELS_DIR / 'based_content_feature_matrix.pkl')
    content_index = joblib.load(MODELS_DIR / 'based_content_index.pkl')
    logging.info("Content-based model and data loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading model files: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error while loading model: {e}")
    raise

@app.route('/recommend', methods=['GET'])
@jwt_required()
def recommend():
    product_id = request.args.get('product_id')
    top_n = int(request.args.get('top_n', 10))

    if not product_id:
        return jsonify({'error': 'product_id parameter is required'}), 400

    try:
        # Validar se o produto está no índice
        product_id = int(product_id)  # Certifique-se de que o ID seja inteiro
        if product_id not in content_index:
            return jsonify({'error': f'Product ID {product_id} not found in index'}), 404

        # Obter recomendações
        item_idx = content_index.get_loc(product_id)
        distances, indices = content_knn_model.kneighbors(
            content_feature_matrix[item_idx].reshape(1, -1),
            n_neighbors=top_n + 1  # +1 para incluir o próprio produto
        )

        # Excluir o próprio produto
        mask = indices.flatten() != item_idx
        recommended_indices = indices.flatten()[mask][:top_n]
        recommended_products = content_index[recommended_indices].tolist()

        return jsonify({
            'product_id': product_id,
            'recommendations': recommended_products
        })
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
