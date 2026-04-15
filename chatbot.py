from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import yfinance as yf
from datetime import datetime
import json
import re
import torch
from model_trainer import ModelTrainer
import logging
import os
from market_analyzer import MarketAnalyzer
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

class FinancialChatbot:
    def __init__(self, sentiment_pipeline=None):
        self.market_analyzer = MarketAnalyzer(sentiment_pipeline=sentiment_pipeline) if sentiment_pipeline else MarketAnalyzer()
        self.reset()
        self.model = None
        self.model_features = None
        self.symbol_map = {
            # Stocks
            'apple': 'AAPL', 'appl': 'AAPL', 'aapl': 'AAPL',
            'tesla': 'TSLA', 'tsla': 'TSLA',
            'intel': 'INTC', 'intc': 'INTC',
            'nvidia': 'NVDA', 'nvda': 'NVDA',
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'meta': 'META', 'facebook': 'META',
            'google': 'GOOGL', 'alphabet': 'GOOGL',
            'amazon': 'AMZN', 'amzn': 'AMZN',
            # Crypto
            'btc': 'BTC-USD', 'bitcoin': 'BTC-USD',
            'eth': 'ETH-USD', 'ethereum': 'ETH-USD',
            'doge': 'DOGE-USD', 'dogecoin': 'DOGE-USD',
            'ltc': 'LTC-USD', 'litecoin': 'LTC-USD',
            'bnb': 'BNB-USD', 'binance': 'BNB-USD',
            'ada': 'ADA-USD', 'cardano': 'ADA-USD',
            'sol': 'SOL-USD', 'solana': 'SOL-USD',
            # Indexes
            'sp500': '^GSPC', 's&p500': '^GSPC', 'nasdaq': '^IXIC', 'dow': '^DJI',
        }
        logging.basicConfig(
            filename=f'chatbot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def reset(self):
        self.state = 'awaiting_symbol'  # or 'awaiting_question'
        self.symbol = None

    def process_crypto_symbol(self, symbol):
        """Converts a crypto symbol to Yahoo Finance format"""
        crypto_mapping = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'XRP': 'XRP-USD'
        }
        # Remove USD if present and check mapping
        clean_symbol = symbol.upper().replace('USD', '')
        if clean_symbol in crypto_mapping:
            return crypto_mapping[clean_symbol]
        return symbol

    def process_message(self, message):
        try:
            # Detecting crypto symbol in message
            crypto_symbols = {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
                'DOGE': 'DOGE-USD',
                'ADA': 'ADA-USD',
                'XRP': 'XRP-USD'
            }
            # Check if message contains a crypto symbol
            for crypto in crypto_symbols:
                if crypto.lower() in message.lower():
                    symbol = crypto_symbols[crypto]
                    # The rest of the processing logic...
                    return f"Analyzing {crypto}... Symbol: {symbol}"
            
            message = message.strip()
            if self.state == 'awaiting_symbol':
                symbol = self._extract_symbol(message)
                if symbol:
                    self.symbol = symbol
                    self.state = 'awaiting_question'
                    return f"You selected symbol: {self.symbol}. What information do you want? (e.g., price, price week, price month, news, trend, recommendation, plot)"
                else:
                    return "Please enter a valid stock or crypto symbol (e.g., NVDA, AAPL, BTC-USD, TSLA, ETH-USD):"
            elif self.state == 'awaiting_question':
                if not self.symbol:
                    self.state = 'awaiting_symbol'
                    return "Please enter a stock or crypto symbol first (e.g., NVDA, AAPL, BTC-USD, TSLA, ETH-USD):"
                lower = message.lower()
                if 'price month' in lower:
                    return self._get_price(period='1mo')
                elif 'price week' in lower:
                    return self._get_price(period='7d')
                elif 'price' in lower:
                    return self._get_price(period='1d')
                elif 'news' in lower:
                    return self._get_news()
                elif 'trend' in lower:
                    return self._get_trend()
                elif 'recommend' in lower or 'recommendation' in lower or 'predict' in lower:
                    return self._get_recommendation()
                elif 'plot' in lower or 'wykres' in lower:
                    return self._get_plot()
                elif 'change symbol' in lower or 'other symbol' in lower or 'reset' in lower:
                    self.reset()
                    return "Okay, let's start over. Please enter a stock or crypto symbol (e.g., NVDA, AAPL, BTC-USD, TSLA, ETH-USD):"
                else:
                    return "Sorry, I didn't understand. Please ask about: price, price week, price month, news, trend, recommendation, predict, plot. Or type 'change symbol' to start over."
            else:
                self.reset()
                return "Let's start over. Please enter a stock or crypto symbol (e.g., NVDA, AAPL, BTC-USD, TSLA, ETH-USD):"
        except Exception as e:
            logging.error(f"Error in process_message: {str(e)}")
            self.reset()
            return "Sorry, something went wrong. Let's start over. Please enter a stock or crypto symbol (e.g., NVDA, AAPL, BTC-USD, TSLA, ETH-USD):"

    def _extract_symbol(self, message):
        msg = message.strip().lower()
        # Try mapping from known names
        for key, val in self.symbol_map.items():
            if key in msg:
                return val
        # Try to extract a valid ticker
        match = re.search(r'\b([A-Z]{1,5}(?:\.WA|-USD)?)\b', message.upper())
        if match:
            symbol = match.group(1)
            # For crypto, enforce -USD
            if symbol in ['BTC', 'ETH', 'DOGE', 'LTC', 'BNB', 'ADA', 'SOL']:
                return symbol + '-USD'
            return symbol
        return None

    def _validate_symbol(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            return not data.empty
        except Exception:
            return False

    def _get_price(self, period='1d'):
        try:
            logging.info(f"Fetching price for symbol: {self.symbol}, period: {period}")
            if not self._validate_symbol(self.symbol):
                return f"No data found for symbol '{self.symbol}'. Please check the symbol and try again."
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period)
            if data.empty:
                return f"No price data found for {self.symbol} (period: {period}). Try another symbol or period."
            if period != '1d':
                closes = data['Close']
                closes_str = '\n'.join([f"{d.strftime('%Y-%m-%d')}: {v:.2f} USD" for d, v in closes.items()])
                return f"Closing prices for {self.symbol} ({period}):\n{closes_str}"
            else:
                price = data['Close'].iloc[-1]
                return f"Current price for {self.symbol}: {price:.2f} USD"
        except Exception as e:
            logging.error(f"Error getting price for {self.symbol} (period: {period}): {str(e)}")
            return f"Could not fetch price for {self.symbol} (period: {period}). Error: {str(e)}"

    def _get_news(self):
        try:
            logging.info(f"Fetching news for symbol: {self.symbol}")
            if not self._validate_symbol(self.symbol):
                return f"No data found for symbol '{self.symbol}'. Please check the symbol and try again."
            news = self.market_analyzer._get_news(self.symbol)
            if not news:
                return f"No news found for {self.symbol}."
            response = f"Latest news for {self.symbol}:\n"
            for art in news[:3]:
                response += f"- {art.get('title', 'No title')}\n"
            return response
        except Exception as e:
            logging.error(f"Error getting news for {self.symbol}: {str(e)}")
            return f"Could not fetch news for {self.symbol}. Error: {str(e)}"

    def _get_trend(self):
        try:
            logging.info(f"Fetching trend for symbol: {self.symbol}")
            if not self._validate_symbol(self.symbol):
                return f"No data found for symbol '{self.symbol}'. Please check the symbol and try again."
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period='1y')
            if data.empty:
                return f"No historical data for {self.symbol}."
            sma20 = SMAIndicator(close=data['Close'], window=20).sma_indicator()
            sma50 = SMAIndicator(close=data['Close'], window=50).sma_indicator()
            rsi = RSIIndicator(close=data['Close'], window=14).rsi()
            macd = MACD(close=data['Close']).macd()
            last_close = data['Close'].iloc[-1]
            last_sma20 = sma20.iloc[-1]
            last_sma50 = sma50.iloc[-1]
            last_rsi = rsi.iloc[-1]
            last_macd = macd.iloc[-1]
            trend = "sideways"
            if last_close > last_sma20 > last_sma50:
                trend = "uptrend"
            elif last_close < last_sma20 < last_sma50:
                trend = "downtrend"
            elif last_sma20 > last_sma50:
                trend = "slight uptrend"
            elif last_sma20 < last_sma50:
                trend = "slight downtrend"
            return (f"Trend for {self.symbol}: {trend}\n"
                    f"Last close: {last_close:.2f} USD\n"
                    f"SMA20: {last_sma20:.2f}, SMA50: {last_sma50:.2f}\n"
                    f"RSI: {last_rsi:.2f}, MACD: {last_macd:.2f}")
        except Exception as e:
            logging.error(f"Error getting trend for {self.symbol}: {str(e)}")
            return f"Could not fetch trend for {self.symbol}. Error: {str(e)}"

    def _get_recommendation(self):
        try:
            logging.info(f"Fetching recommendation for symbol: {self.symbol}")
            if not self._validate_symbol(self.symbol):
                return f"No data found for symbol '{self.symbol}'. Please check the symbol and try again."
            data = self.market_analyzer.get_market_data(self.symbol)
            ta = data['technical_analysis']
            pred = data['market_prediction']
            ml_metrics = data.get('ml_metrics')
            last_close = data['current_price']
            last_sma20 = ta.get('SMA20', '?') if isinstance(ta, dict) else '?'
            last_sma50 = ta.get('SMA50', '?') if isinstance(ta, dict) else '?'
            last_rsi = ta.get('RSI', '?') if isinstance(ta, dict) else '?'
            last_macd = ta.get('MACD', '?') if isinstance(ta, dict) else '?'
            rec = ta['recommendation']
            pred_str = 'up' if pred.get('prediction') == 1 else 'down' if pred.get('prediction') == 0 else 'no prediction'
            proba = pred.get('probability') or pred.get('proba') or pred.get('prob', '?')
            # --- METRYKI ---
            if ml_metrics:
                metrics_str = (f"\nModel accuracy: {ml_metrics['accuracy']:.2%}, "
                               f"precision: {ml_metrics['precision']:.2%}, "
                               f"recall: {ml_metrics['recall']:.2%}, "
                               f"f1: {ml_metrics['f1']:.2%}")
            else:
                metrics_str = "\nModel ML nie został wytrenowany – brak metryk dokładności."
            return (f"Recommendation for {self.symbol}: {rec}\n"
                    f"ML prediction for next week: {pred_str}"
                    f"{f' (probability: {proba:.2f})' if isinstance(proba, float) else ''}\n"
                    f"Last close: {last_close:.2f} USD\n"
                    f"{metrics_str}")
        except Exception as e:
            logging.error(f"Error getting recommendation for {self.symbol}: {str(e)}")
            return f"Could not fetch recommendation for {self.symbol}. Error: {str(e)}"

    def _initialize_model(self, model_path):
        """Inicjalizacja modelu językowego - obecnie wyłączona"""
        pass  # Nie inicjalizujemy modelu, używamy tylko logiki bazowej

    def _generate_model_response(self, message):
        """Generowanie odpowiedzi za pomocą modelu językowego"""
        try:
            # Przygotowanie promptu
            prompt = f"input: {message}\noutput:"
            
            # Tokenizacja
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generowanie odpowiedzi
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Dekodowanie odpowiedzi
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Wyciąganie tylko części odpowiedzi
            response = response.split("output:")[-1].strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Błąd podczas generowania odpowiedzi modelu: {str(e)}")
            return None

    def _generate_standard_response(self, message):
        """Generowanie standardowej odpowiedzi (poprzednia logika)"""
        message = message.lower().strip()
        query_type = self._identify_query_type(message)
        
        # Check if we have symbols in context
        symbols = self._extract_symbols(message)
        
        if query_type == "investment_recommendation":
            if symbols:
                return self._generate_investment_recommendation(message)
            return "Mogę pomóc Ci z rekomendacjami inwestycyjnymi. Czy możesz sprecyzować, które akcje lub sektory Cię interesują?"
        elif query_type == "market_analysis":
            return self._analyze_market(message)
        elif query_type == "stock_info":
            if symbols:
                return self._get_stock_info(message)
            return "Proszę podać, o której spółce chciałbyś/chciałabyś uzyskać informacje."
        elif query_type == "greeting":
            return self._generate_greeting()
        else:
            # If we have symbols in context, use them
            if symbols:
                return self._get_stock_info(message)
            return self._generate_general_response(message)

    def train_model(self, num_epochs=3, batch_size=4):
        """Trenowanie modelu na zebranych konwersacjach"""
        try:
            # Ładowanie konwersacji
            conversations = self.model_trainer.load_conversations()
            
            if not conversations:
                logging.warning("Brak konwersacji do treningu")
                return False
            
            # Przygotowanie danych treningowych
            self.model_trainer.prepare_training_data(conversations)
            
            # Trenowanie modelu
            self.model_trainer.train(num_epochs=num_epochs, batch_size=batch_size)
            
            # Aktualizacja modelu
            self._initialize_model("./trained_model")
            
            logging.info("Model pomyślnie wytrenowany")
            return True
            
        except Exception as e:
            logging.error(f"Błąd podczas trenowania modelu: {str(e)}")
            return False

    def _identify_query_type(self, message):
        """Identyfikacja typu zapytania z wiadomości użytkownika"""
        message = message.lower().strip()
        
        # Najpierw sprawdź czy mamy symbole w wiadomości
        symbols = self._extract_symbols(message)
        if symbols:
            return "stock_info"
        
        # Wzorce dla różnych typów zapytań
        investment_patterns = [
            r'inwest', r'kup', r'sprzedaj', r'warto', r'rekomend',
            r'okazj', r'dobra inwestycj', r'akcj[ęe] do kupna',
            r'co kupi[ćc]', r'gdzie zainwestowa[ćc]', r'co s[ąa]dzisz o',
            r'jaka jest sytuacja', r'jak wygl[ąa]da', r'analiza'
        ]
        
        market_patterns = [
            r'rynek', r'trend', r'analiz', r'perspektyw',
            r'prognoz', r'przewidywani', r'co si[ęe] dzieje',
            r'jak wygl[ąa]da', r'sektor'
        ]
        
        stock_patterns = [
            r'akcj', r'udzia[łl]', r'cena', r'warto[śs][ćc]',
            r'informacje o', r'szczeg[óo][łl]y', r'co s[ąa]dzisz o',
            r'jaka jest sytuacja', r'jak wygl[ąa]da'
        ]
        
        greeting_patterns = [
            r'cze[śs][ćc]', r'witaj', r'dzie[ńn] dobry',
            r'dobry wiecz[óo]r', r'dobry poranek', r'hej'
        ]
        
        # Sprawdź wzorce w kontekście całej wiadomości
        if any(re.search(pattern, message) for pattern in investment_patterns):
            # Jeśli wiadomość zawiera nazwę spółki lub symbol, traktuj jako zapytanie o spółkę
            if any(company in message for company in self.company_to_symbol.keys()):
                return "stock_info"
            return "investment_recommendation"
        elif any(re.search(pattern, message) for pattern in market_patterns):
            return "market_analysis"
        elif any(re.search(pattern, message) for pattern in stock_patterns):
            # Jeśli wiadomość zawiera nazwę spółki lub symbol, traktuj jako zapytanie o spółkę
            if any(company in message for company in self.company_to_symbol.keys()):
                return "stock_info"
            return "stock_info"
        elif any(re.search(pattern, message) for pattern in greeting_patterns):
            return "greeting"
        else:
            # Sprawdź czy wiadomość zawiera nazwę spółki
            if any(company in message for company in self.company_to_symbol.keys()):
                return "stock_info"
            return "general"
    
    def _generate_investment_recommendation(self, message):
        """Generate investment recommendations based on market analysis"""
        try:
            symbols = self._extract_symbols(message)
            if not symbols:
                return "Mogę pomóc Ci z rekomendacjami inwestycyjnymi. Czy możesz sprecyzować, które akcje lub sektory Cię interesują?"
            responses = []
            for symbol in symbols[:3]:
                try:
                    data = self.market_analyzer.get_market_data(symbol)
                    ta = data['technical_analysis']
                    pred = data['market_prediction']
                    news = data['news']
                    company = data['company_info']
                    ml_metrics = data.get('ml_metrics')
                    if ml_metrics:
                        metrics_str = (f"\nModel accuracy: {ml_metrics['accuracy']:.2%}, "
                                       f"precision: {ml_metrics['precision']:.2%}, "
                                       f"recall: {ml_metrics['recall']:.2%}, "
                                       f"f1: {ml_metrics['f1']:.2%}")
                    else:
                        metrics_str = "\nModel ML nie został wytrenowany – brak metryk dokładności."
                    response = f"\n---\nSpółka: {company['name']} ({symbol})\nSektor: {company['sector']} | Branża: {company['industry']}\nKapitalizacja: {company['market_cap']:,}\n\n" \
                        f"Analiza techniczna:\n- Trend: {ta['trend']}\n- Momentum (RSI): {ta['momentum']}\n- MACD: {ta['macd']}\n- Stochastic: {ta['stochastic']}\n- Zmienność: {ta['volatility']}\n- Rekomendacja: {ta['recommendation']}\n" \
                        f"\nPredykcja ML: {'Wzrost' if pred['prediction']==1 else 'Spadek' if pred['prediction']==0 else 'Brak predykcji'}{metrics_str}\n" \
                        f"\nNajnowsze newsy:\n"
                    for art in news:
                        response += f"- {art['title']} (Sentyment: {art.get('sentiment','?')}, Score: {art.get('sentiment_score',0):.2f})\n"
                    responses.append(response)
                except Exception as e:
                    responses.append(f"Błąd analizy dla {symbol}: {str(e)}")
            return "\n".join(responses)
        except Exception as e:
            return "Przepraszam, ale mam problem z generowaniem rekomendacji inwestycyjnych. Proszę spróbować ponownie później."
    
    def _analyze_market(self, message):
        """Analyze market conditions and provide insights"""
        try:
            # Przykład: domyślnie S&P500
            data = self.market_analyzer.get_market_data('^GSPC')
            ta = data['technical_analysis']
            pred = data['market_prediction']
            news = data['news']
            response = f"Indeks S&P 500:\n- Trend: {ta['trend']}\n- Momentum (RSI): {ta['momentum']}\n- MACD: {ta['macd']}\n- Stochastic: {ta['stochastic']}\n- Zmienność: {ta['volatility']}\n- Rekomendacja: {ta['recommendation']}\n" \
                f"\nPredykcja ML: {'Wzrost' if pred['prediction']==1 else 'Spadek' if pred['prediction']==0 else 'Brak predykcji'}\n" \
                f"\nNajnowsze newsy:\n"
            for art in news:
                response += f"- {art['title']} (Sentyment: {art.get('sentiment','?')}, Score: {art.get('sentiment_score',0):.2f})\n"
            return response
        except Exception as e:
            return f"Przepraszam, ale mam problem z analizą rynku: {str(e)}"
    
    def _get_stock_info(self, message):
        """Pobiera i analizuje informacje o akcjach"""
        try:
            symbols = self.conversation_context['last_symbols'] or self._extract_symbols(message)
            if not symbols:
                return "Przepraszam, nie znalazłem informacji o żadnych akcjach. Czy mógłbyś/mogłabyś podać symbol lub nazwę spółki?"
            
            responses = []
            for symbol in symbols:
                try:
                    data = self.market_analyzer.get_market_data(symbol)
                    if data:
                        analysis = self.market_analyzer.analyze_market_data(data)
                        sentiment = self.market_analyzer.analyze_sentiment(symbol)
                        
                        response = f"Analiza dla {symbol}:\n"
                        response += f"• Aktualna cena: {data['current_price']:.2f} {data['currency']}\n"
                        response += f"• Zmiana dzienna: {data['daily_change']:.2f}% ({data['daily_change_amount']:.2f} {data['currency']})\n"
                        
                        if analysis['trend'] == 'up':
                            response += "• Trend: Wzrostowy 📈\n"
                        elif analysis['trend'] == 'down':
                            response += "• Trend: Spadkowy 📉\n"
                        else:
                            response += "• Trend: Boczny ↔️\n"
                            
                        if sentiment['sentiment'] == 'positive':
                            response += "• Sentyment rynku: Pozytywny 😊\n"
                        elif sentiment['sentiment'] == 'negative':
                            response += "• Sentyment rynku: Negatywny 😟\n"
                        else:
                            response += "• Sentyment rynku: Neutralny 😐\n"
                            
                        responses.append(response)
                except Exception as e:
                    logging.error(f"Błąd podczas analizy {symbol}: {str(e)}")
                    responses.append(f"Przepraszam, nie udało się pobrać pełnych informacji dla {symbol}.")
            
            return "\n\n".join(responses)
            
        except Exception as e:
            logging.error(f"Błąd podczas pobierania informacji o akcjach: {str(e)}")
            return "Przepraszam, wystąpił problem podczas analizy akcji. Spróbuj ponownie później."

    def _translate_trend(self, trend):
        """Tłumaczenie trendu na język polski"""
        translations = {
            'Bullish trend': 'Trend wzrostowy',
            'Bearish trend': 'Trend spadkowy',
            'Neutral trend': 'Trend boczny'
        }
        return translations.get(trend, trend)

    def _translate_momentum(self, momentum):
        """Tłumaczenie momentum na język polski"""
        translations = {
            'Overbought': 'Wykupiony',
            'Oversold': 'Wyprzedany',
            'Neutral': 'Neutralny'
        }
        return translations.get(momentum, momentum)

    def _translate_macd(self, macd):
        """Tłumaczenie MACD na język polski"""
        translations = {
            'Bullish': 'Wzrostowy',
            'Bearish': 'Spadkowy',
            'Neutral': 'Neutralny'
        }
        return translations.get(macd, macd)

    def _translate_stochastic(self, stoch):
        """Tłumaczenie Stochastic na język polski"""
        translations = {
            'Overbought': 'Wykupiony',
            'Oversold': 'Wyprzedany',
            'Neutral': 'Neutralny'
        }
        return translations.get(stoch, stoch)

    def _translate_volatility(self, vol):
        """Tłumaczenie zmienności na język polski"""
        translations = {
            'High': 'Wysoka',
            'Low': 'Niska',
            'Medium': 'Średnia'
        }
        return translations.get(vol, vol)

    def _translate_recommendation(self, rec):
        """Tłumaczenie rekomendacji na język polski"""
        translations = {
            'Strong Buy': 'Silny sygnał kupna',
            'Buy': 'Sygnał kupna',
            'Hold': 'Trzymaj',
            'Sell': 'Sygnał sprzedaży',
            'Strong Sell': 'Silny sygnał sprzedaży'
        }
        return translations.get(rec, rec)

    def _translate_prediction(self, pred):
        """Tłumaczenie predykcji na język polski"""
        translations = {
            1: 'Wzrost',
            0: 'Spadek',
            -1: 'Brak predykcji'
        }
        return translations.get(pred, 'Brak predykcji')

    def _translate_sentiment(self, sentiment):
        """Tłumaczenie sentymentu na język polski"""
        translations = {
            'POSITIVE': 'Pozytywny',
            'NEGATIVE': 'Negatywny',
            'NEUTRAL': 'Neutralny'
        }
        return translations.get(sentiment, sentiment)
    
    def _generate_greeting(self):
        """Generuje powitanie tylko przy pierwszej wiadomości"""
        if not self.conversation_context['greeting_sent']:
            self.conversation_context['greeting_sent'] = True
            return "Witaj! Jestem Twoim AI doradcą finansowym. Mogę pomóc Ci w analizie akcji, kryptowalut i innych aktywów. O czym chciałbyś/chciałabyś porozmawiać?"
        return None

    def _generate_follow_up(self, query_type, symbols=None):
        """Generuje pytanie uzupełniające bazując na kontekście"""
        if query_type == "stock_info" and symbols:
            return f"Czy chciałbyś/chciałabyś dowiedzieć się czegoś więcej o {', '.join(symbols)}? Na przykład o ich wynikach finansowych, prognozach czy analizie technicznej?"
        elif query_type == "investment_recommendation":
            return "Czy chciałbyś/chciałabyś, żebym skupił się na jakimś konkretnym sektorze lub typie aktywów?"
        elif query_type == "market_analysis":
            return "Czy interesuje Cię jakiś konkretny aspekt rynku? Na przykład trendy, wolumen czy zmienność?"
        return None
    
    def _generate_general_response(self, message):
        """Generowanie ogólnej odpowiedzi w języku polskim"""
        return "Mogę pomóc Ci w analizie różnych aktywów finansowych. Możesz zapytać mnie o:\n" \
               f"- Konkretne spółki (np. 'Co sądzisz o Nvidii?')\n" \
               f"- Kryptowaluty (np. 'Jaka jest sytuacja Bitcoina?')\n" \
               f"- ETF-y (np. 'Jak wygląda ETF ARKK?')\n" \
               f"- Trendy rynkowe (np. 'Jak wygląda sektor technologiczny?')\n\n" \
               f"Co Cię konkretnie interesuje?"
    
    def _extract_symbols(self, message):
        """Wyciąga symbole giełdowe z wiadomości"""
        symbols = []
        
        # Najpierw sprawdź mapowanie nazw firm
        message_lower = message.lower()
        for company, symbol in self.company_to_symbol.items():
            if company in message_lower:
                symbols.append(symbol)
        
        # Jeśli nie znaleziono w mapowaniu, szukaj symboli bezpośrednio
        if not symbols:
            # Szukaj symboli w formacie standardowym (np. NVDA, AAPL)
            symbol_pattern = r'\b[A-Z]{1,5}(?:-[A-Z]{3})?\b'
            found_symbols = re.findall(symbol_pattern, message.upper())
            symbols.extend(found_symbols)
        
        # Jeśli nadal nie znaleziono symboli, sprawdź kontekst
        if not symbols and self.conversation_context['last_symbols']:
            symbols = self.conversation_context['last_symbols']
        
        return list(set(symbols))  # Usuń duplikaty 

    def _get_plot(self):
        try:
            import os
            indicators = ['SMA_20', 'EMA_20', 'ATR_14', 'ADX_14', 'CCI_20', 'OBV']
            filename = f"plot_{self.symbol}.png"
            path = self.market_analyzer.generate_plot(self.symbol, period='3mo', indicators=indicators, output_file=filename)
            if path and os.path.exists(path):
                return f"Plot generated: {path} (open this file to view the chart)"
            elif path is None:
                return "Not enough historical data to generate a plot with advanced indicators (need at least 20 days)."
            else:
                return "Could not generate plot."
        except Exception as e:
            logging.error(f"Error generating plot for {self.symbol}: {str(e)}")
            return f"Could not generate plot for {self.symbol}. Error: {str(e)}"