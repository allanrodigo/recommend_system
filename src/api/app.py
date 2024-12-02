import sys
from os import getcwd
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required
from flask_cors import CORS
from pathlib import Path
from decouple import config
import logging

print("PYTHONPATH:", sys.path)
print("Current Directory:", getcwd())

# Adicionar o diretório raiz ao caminho
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.recommender.content_filtering import generate_recommendations, load_model
from config import setup_logging

setup_logging()

model = load_model()

# Inicialização do Flask
app = Flask(__name__)
CORS(app)

# Configuração do JWT
app.config['JWT_SECRET_KEY'] = config("JWT_SECRET_KEY")
jwt = JWTManager(app)

@app.route('/recommend', methods=['GET'])
#@jwt_required()
def recommend():
    """
    Endpoint para gerar recomendações.
    """
    try:
        product_id = int(request.args.get('product_id'))
        top_n = int(request.args.get('top_n', 10))
        if not product_id:
            return jsonify({'error': 'O parâmetro product_id é obrigatório.'}), 400

        response = generate_recommendations(product_id, top_n=top_n, model=model)

        if not response:
            return jsonify({'recommendations': []})

        return jsonify(response)

    except ValueError as e:
        logging.warning(f"Erro de validação: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Erro interno no servidor: {e}")
        return jsonify({'error': 'Erro interno no servidor.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    