import pytest
from src.recommender.hybrid_filtering import HybridRecommender
from src.api.app import app

@pytest.fixture
def client():
    # Sets up the Flask test client
    with app.test_client() as client:
        yield client

def test_load_model():
    hybrid_recommender = HybridRecommender.load_model()
    assert hybrid_recommender is not None, "O modelo deveria ser carregado com sucesso."

def test_recommendations_endpoint(client):
    product_id = '1801805'
    response = client.get(f"/recommend/{product_id}")

    # Check that the request was successful (status code 200)
    assert response.status_code == 200

    # Check that the response contains JSON data
    assert response.is_json

    # Check the structure of the JSON response (assuming it's a list of IDs)
    data = response.get_json()
    assert isinstance(data, list), "Expected a list of recommended product IDs"
    assert all(isinstance(item, str) for item in data), "Each recommendation should be a string product ID"

    # Assuming top_n is 10 by default
    assert len(data) == 10, "Expected 10 recommendations by default"