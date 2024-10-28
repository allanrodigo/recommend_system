from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required
from flask_cors import CORS
from recommender.hybrid_filtering import HybridRecommender
from config import API_KEY

app = Flask(__name__)
CORS(app)

# JWT config
app.config['SECRET_KEY'] = API_KEY
jwt = JWTManager(app)

hybrid_recommender = HybridRecommender.load_model()

@app.route('/recommend', methods=['GET'])
@jwt_required()
def recommend():
    product_id = request.args.get('product_id')
    top_n = int(request.args.get('top_n', 10))

    if not product_id:
        return jsonify({'error': 'product_id parameter is required'}), 400

    # get recommendations
    recommendations = hybrid_recommender.recommend(
        product_id=product_id,
        top_n=top_n,
    )

    if not recommendations:
        return jsonify({'error': f'No recommendations found for product_id {product_id}'}), 404

    return jsonify({'product_id': product_id, 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(host='192.168.0.100', port=8000, debug=False)