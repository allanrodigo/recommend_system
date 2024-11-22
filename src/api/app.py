import sys
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required
from flask_cors import CORS
from pathlib import Path
import logging
import joblib

# Adicionar o diretório raiz ao caminho
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import API_KEY  
from src.recommender.content_filtering import generate_recommendations, load_model

# Inicialização do Flask
app = Flask(__name__)
CORS(app)

# Configuração do JWT
app.config['API_KEY'] = API_KEY
app.config['JWT_SECRET_KEY'] = API_KEY
jwt = JWTManager(app)

# Diretórios
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# Carregar o modelo baseado em conteúdo e seus dados auxiliares
try:
    content_index = load_model('based_content_index.pkl')
    logging.info("Content index loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading model files: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error while loading model: {e}")
    raise

@app.route('/recommend', methods=['GET'])
#@jwt_required()
def recommend():
    """
    Endpoint para gerar recomendações.
    """
    product_id = request.args.get('product_id')
    top_n = int(request.args.get('top_n', 10))

    # Validações
    if not product_id:
        return jsonify({'error': 'product_id parameter is required'}), 400
    if not product_id.isdigit():
        return jsonify({'error': 'product_id must be a valid integer'}), 400

    try:
        product_id = int(product_id)
        recommendations = generate_recommendations(product_id, top_n=top_n)
        
        if product_id not in content_index:
            logging.warning(f"Product ID {product_id} not found in index.")
            return jsonify({'error': f'Product ID {product_id} not found in index'}), 404

        logging.info(f"Recommendations for product {product_id}: {recommendations}")
        return jsonify({
            'product_id': product_id,
            'recommendations': recommendations
        })

    except ValueError as e:
        logging.warning(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logging.error(f"Internal server error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Use debug=True apenas em desenvolvimento
    app.run(host='0.0.0.0', port=8000, debug=True)
