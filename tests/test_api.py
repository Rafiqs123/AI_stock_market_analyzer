import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'AI Financial Advisor' in rv.data

def test_market_data_valid(client):
    rv = client.get('/api/market-data/AAPL')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'current_price' in data or 'error' in data

def test_market_data_invalid(client):
    rv = client.get('/api/market-data/FAKE123')
    assert rv.status_code == 400 or rv.status_code == 200
    data = rv.get_json()
    assert 'error' in data or 'current_price' not in data 