import sys
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required
from flask_cors import CORS
from pathlib import Path
import logging

# Adicionar o diretório raiz ao caminho
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import API_KEY
from src.recommender.content_filtering import generate_recommendations

# Inicialização do Flask
app = Flask(__name__)
CORS(app)

# Configuração do JWT
app.config['API_KEY'] = API_KEY
app.config['JWT_SECRET_KEY'] = API_KEY
jwt = JWTManager(app)

@app.route('/recommend', methods=['GET'])
#@jwt_required()
def recommend():
    """
    Endpoint para gerar recomendações.
    """
    try:
        product_id = request.args.get('product_id')
        top_n = int(request.args.get('top_n', 10))

        # Validações
        if not product_id:
            return jsonify({'error': 'O parâmetro product_id é obrigatório.'}), 400
        if not product_id.isdigit():
            return jsonify({'error': 'O product_id deve ser um número inteiro válido.'}), 400

        # Gerar recomendações usando o método direto
        recommendations = generate_recommendations(int(product_id), top_n=top_n)

        return jsonify({
            'product_id': product_id,
            'recommendations': recommendations
        })

    except ValueError as e:
        logging.warning(f"Erro de validação: {e}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logging.error(f"Erro interno no servidor: {e}")
        return jsonify({'error': 'Erro interno no servidor.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)