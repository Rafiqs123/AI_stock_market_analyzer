import pytest
from chatbot import FinancialChatbot

@pytest.fixture
def chatbot():
    return FinancialChatbot()

def test_extract_symbol(chatbot):
    assert chatbot._extract_symbol('apple') == 'AAPL'
    assert chatbot._extract_symbol('NVDA') == 'NVDA'
    assert chatbot._extract_symbol('btc') == 'BTC-USD'
    assert chatbot._extract_symbol('ETH') == 'ETH-USD'
    assert chatbot._extract_symbol('tesla') == 'TSLA'
    assert chatbot._extract_symbol('unknown') is None

def test_validate_symbol(chatbot):
    assert chatbot._validate_symbol('AAPL') is True
    assert chatbot._validate_symbol('NVDA') is True
    assert chatbot._validate_symbol('BTC-USD') is True
    assert chatbot._validate_symbol('FAKE123') is False

def test_process_message_symbol(chatbot):
    response = chatbot.process_message('apple')
    assert 'AAPL' in response or 'symbol' in response.lower()

def test_process_message_price(chatbot):
    chatbot.state = 'awaiting_question'
    chatbot.symbol = 'AAPL'
    response = chatbot.process_message('price')
    assert 'Current price' in response or 'No data found' in response

def test_process_message_recommendation(chatbot):
    chatbot.state = 'awaiting_question'
    chatbot.symbol = 'AAPL'
    response = chatbot.process_message('recommendation')
    assert 'Recommendation' in response or 'No data found' in response or 'Could not fetch' in response 