import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
import requests
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
import functools
import redis
import pickle
from ta.volume import OnBalanceVolumeIndicator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import time

# Try to import ratelimit, provide mock if not available
try:
    from ratelimit import limits, sleep_and_retry
    RATELIMIT_AVAILABLE = True
except ImportError:
    logging.warning("ratelimit package not installed. Rate limiting will be disabled.")
    RATELIMIT_AVAILABLE = False
    
    # Mock decorators
    def sleep_and_retry(func):
        return func
        
    def limits(calls=None, period=None):
        def decorator(func):
            return func
        return decorator

from functools import lru_cache

load_dotenv()

class MarketAnalyzer:
    def __init__(self, sentiment_pipeline=None):
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        if sentiment_pipeline is not None:
            self.sentiment_analyzer = sentiment_pipeline
        else:
            from transformers import pipeline
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.model_path = "market_predictor.pkl"
        self.model = self._load_predictor()
        self._news_sentiment_cache = {}  # cache for the process lifetime
        # Redis setup
        try:
            self.redis = redis.Redis(host='localhost', port=6379, db=0)
            self.redis.ping()
        except Exception:
            self.redis = None
        self._local_cache = {}
        # Mapping of popular cryptocurrencies
        self.crypto_mapping = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'DOGE': 'DOGE-USD',
            'ADA': 'ADA-USD',
            'DOT': 'DOT-USD',
            'XRP': 'XRP-USD',
            # add more pairs as needed
        }

    def _cache_set(self, key, value, ex=600):
        try:
            if self.redis:
                self.redis.set(key, pickle.dumps(value), ex=ex)
            else:
                self._local_cache[key] = (value, datetime.now(), ex)
        except Exception:
            self._local_cache[key] = (value, datetime.now(), ex)

    def _cache_get(self, key):
        try:
            if self.redis:
                val = self.redis.get(key)
                if val:
                    return pickle.loads(val)
            else:
                v = self._local_cache.get(key)
                if v:
                    value, ts, ex = v
                    if (datetime.now() - ts).total_seconds() < ex:
                        return value
                    else:
                        del self._local_cache[key]
        except Exception:
            return None
        return None
        
    @sleep_and_retry
    @limits(calls=2, period=1)  # Maximum 2 calls per second
    @lru_cache(maxsize=100)
    def get_market_data(self, symbol, period='1y'):
        """Fetch market data for a given symbol, with cache"""
        cache_key = f"market:{symbol}:{period}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            # Market data always from a longer period (default 1y or 1mo)
            formatted_symbol = self.format_symbol(symbol)
            ticker = yf.Ticker(formatted_symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                raise Exception(f"No historical data for {symbol}")
            hist['SMA_20'] = SMAIndicator(close=hist['Close'], window=20).sma_indicator()
            hist['EMA_20'] = EMAIndicator(close=hist['Close'], window=20).ema_indicator()
            hist['RSI'] = RSIIndicator(close=hist['Close']).rsi()
            macd = MACD(close=hist['Close'])
            hist['MACD'] = macd.macd()
            hist['MACD_signal'] = macd.macd_signal()
            hist['MACD_diff'] = macd.macd_diff()
            stoch = StochasticOscillator(high=hist['High'], low=hist['Low'], close=hist['Close'])
            hist['Stoch'] = stoch.stoch()
            hist['Stoch_signal'] = stoch.stoch_signal()
            bb = BollingerBands(close=hist['Close'])
            hist['BB_upper'] = bb.bollinger_hband()
            hist['BB_lower'] = bb.bollinger_lband()
            # --- New indicators ---
            hist['ATR_14'] = AverageTrueRange(high=hist['High'], low=hist['Low'], close=hist['Close'], window=14).average_true_range()
            hist['ADX_14'] = ADXIndicator(high=hist['High'], low=hist['Low'], close=hist['Close'], window=14).adx()
            hist['CCI_20'] = CCIIndicator(high=hist['High'], low=hist['Low'], close=hist['Close'], window=20).cci()
            hist['OBV'] = OnBalanceVolumeIndicator(close=hist['Close'], volume=hist['Volume']).on_balance_volume()
            # --- Volume analysis ---
            hist['Volume_Change'] = hist['Volume'].pct_change()
            volume_alert = False
            if len(hist) > 2 and abs(hist['Volume_Change'].iloc[-1]) > 1.0:
                volume_alert = True
            # --- Add news sentiment ---
            # Limit news to last 7 days, but do not limit market data!
            last_dates = hist.index[-7:] if len(hist) > 7 else hist.index
            news_sentiments = self._get_daily_news_sentiment(symbol, last_dates.strftime('%Y-%m-%d'))
            hist['news_sentiment'] = 0.0
            hist.loc[last_dates, 'news_sentiment'] = news_sentiments
            # --- ML prediction ---
            ml_pred_dict = self._predict_market_movement(hist)
            ml_pred = ml_pred_dict.get('prediction')
            ml_proba = None
            ml_metrics = ml_pred_dict.get('metrics')
            market_data = {
                'current_price': hist['Close'].iloc[-1],
                'currency': ticker.info.get('currency', 'USD'),
                'daily_change': ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0,
                'daily_change_amount': hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0,
                'company_info': self._prepare_company_info(ticker.info, symbol),
                'technical_analysis': self._analyze_technical_indicators(hist, ml_pred, ml_proba),
                'market_prediction': ml_pred_dict,
                'ml_metrics': ml_metrics,
                'news': self._get_news(symbol),
                'atr': hist['ATR_14'].iloc[-1],
                'adx': hist['ADX_14'].iloc[-1],
                'cci': hist['CCI_20'].iloc[-1],
                'obv': hist['OBV'].iloc[-1],
                'volume_alert': volume_alert
            }
            self._cache_set(cache_key, market_data, ex=300)  # 5 minut
            return market_data
        except Exception as e:
            logging.error(f"Błąd podczas pobierania danych rynkowych: {str(e)}")
            raise
    
    def _prepare_company_info(self, info, symbol):
        """Przygotowanie informacji o spółce/kryptowalucie"""
        company_info = {
            'name': info.get('longName', info.get('shortName', symbol)),
            'symbol': symbol
        }
        
        # Dla kryptowalut
        if symbol.endswith('-USD'):
            company_info.update({
                'type': 'Kryptowaluta',
                'market_cap': info.get('marketCap', 0),
                'volume_24h': info.get('volume24Hr', 0),
                'circulating_supply': info.get('circulatingSupply', 0),
                'max_supply': info.get('maxSupply', 0)
            })
        # Dla ETF-ów
        elif symbol.startswith(('^', 'SPY', 'QQQ', 'IWM', 'VOO', 'VTI', 'ARK')):
            company_info.update({
                'type': 'ETF',
                'category': info.get('category', 'Nieznana'),
                'total_assets': info.get('totalAssets', 0),
                'expense_ratio': info.get('annualReportExpenseRatio', 0)
            })
        # Dla polskich spółek
        elif symbol.endswith('.WA'):
            company_info.update({
                'type': 'Spółka GPW',
                'sector': info.get('sector', 'Nieznany'),
                'industry': info.get('industry', 'Nieznana'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            })
        # Dla standardowych spółek
        else:
            company_info.update({
                'type': 'Spółka',
                'sector': info.get('sector', 'Nieznany'),
                'industry': info.get('industry', 'Nieznana'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            })
        
        return company_info

    def _get_news(self, symbol, from_param=None, to=None):
        """Pobieranie wiadomości dla danego symbolu, opcjonalnie w zakresie dat, z cache"""
        cache_key = f"news:{symbol}:{from_param}:{to}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            if symbol.endswith('.WA'):
                query = symbol.replace('.WA', '')
                lang = 'pl'
            elif symbol.endswith('-USD'):
                query = symbol.replace('-USD', '')
                lang = 'en'
            elif symbol.startswith('^'):
                query = symbol.replace('^', '')
                lang = 'en'
            else:
                query = symbol
                lang = 'en'
            params = {
                'q': query,
                'language': lang,
                'sort_by': 'publishedAt',
                'page_size': 5
            }
            if from_param:
                params['from_param'] = from_param
            if to:
                params['to'] = to
            news = self.newsapi.get_everything(**params)
            articles = news.get('articles', [])
            self._cache_set(cache_key, articles, ex=600)  # 10 minut
            return articles
        except Exception as e:
            logging.error(f"Błąd podczas pobierania wiadomości: {str(e)}")
            return []
    
    def _get_daily_news_sentiment(self, symbol, dates):
        """Zwraca listę średnich sentymentów newsów dla podanych dat (max 7 zapytań)"""
        sentiments = []
        # Ogranicz do ostatnich 7 dni (by nie przekroczyć limitu NewsAPI)
        if len(dates) > 7:
            dates = dates[-7:]
        for date in dates:
            cache_key = f"{symbol}_{date}"
            if cache_key in self._news_sentiment_cache:
                sentiments.append(self._news_sentiment_cache[cache_key])
                continue
            news = self._get_news(symbol, from_param=date, to=date)
            scores = []
            for art in news:
                try:
                    sent = self.sentiment_analyzer(art.get('title',''))[0]
                    score = sent['score'] if sent['label'].upper() == 'POSITIVE' else -sent['score']
                    scores.append(score)
                except Exception:
                    continue
            avg_sent = float(np.mean(scores)) if scores else 0.0
            self._news_sentiment_cache[cache_key] = avg_sent
            sentiments.append(avg_sent)
        return sentiments

    def _analyze_technical_indicators(self, hist, ml_pred=None, ml_proba=None):
        """Analiza wskaźników technicznych + ML"""
        analysis = {
            'trend': self._analyze_trend(hist),
            'momentum': self._analyze_momentum(hist),
            'volatility': self._analyze_volatility(hist),
            'macd': self._analyze_macd(hist),
            'stochastic': self._analyze_stochastic(hist),
            'atr': hist['ATR_14'].iloc[-1],
            'adx': hist['ADX_14'].iloc[-1],
            'cci': hist['CCI_20'].iloc[-1],
            'obv': hist['OBV'].iloc[-1],
            'volume_alert': abs(hist['Volume_Change'].iloc[-1]) > 1.0 if len(hist) > 2 else False,
            'recommendation': self._generate_recommendation(hist, ml_pred, ml_proba)
        }
        return analysis
    
    def _analyze_trend(self, hist):
        """Analiza trendu cenowego"""
        last_price = hist['Close'].iloc[-1]
        sma_20 = hist['SMA_20'].iloc[-1]
        ema_20 = hist['EMA_20'].iloc[-1]
        
        if last_price > sma_20 and last_price > ema_20:
            return "Trend wzrostowy"
        elif last_price < sma_20 and last_price < ema_20:
            return "Trend spadkowy"
        else:
            return "Trend boczny"
    
    def _analyze_momentum(self, hist):
        """Analiza momentum używając RSI"""
        rsi = hist['RSI'].iloc[-1]
        
        if rsi > 70:
            return "Wykupiony"
        elif rsi < 30:
            return "Wyprzedany"
        else:
            return "Neutralny"
    
    def _analyze_volatility(self, hist):
        """Analiza zmienności"""
        bb_width = (hist['BB_upper'].iloc[-1] - hist['BB_lower'].iloc[-1]) / hist['Close'].iloc[-1]
        
        if bb_width > 0.05:  # 5% szerokości
            return "Wysoka"
        elif bb_width < 0.02:  # 2% szerokości
            return "Niska"
        else:
            return "Średnia"
    
    def _analyze_macd(self, hist):
        """Analiza MACD"""
        macd = hist['MACD'].iloc[-1]
        signal = hist['MACD_signal'].iloc[-1]
        
        if macd > signal and macd > 0:
            return "Wzrostowy"
        elif macd < signal and macd < 0:
            return "Spadkowy"
        else:
            return "Neutralny"
    
    def _analyze_stochastic(self, hist):
        """Analiza Stochastic Oscillator"""
        stoch = hist['Stoch'].iloc[-1]
        
        if stoch > 80:
            return "Wykupiony"
        elif stoch < 20:
            return "Wyprzedany"
        else:
            return "Neutralny"
    
    def _generate_recommendation(self, hist, ml_pred=None, ml_proba=None):
        """Generowanie rekomendacji na podstawie wszystkich wskaźników oraz predykcji ML"""
        trend = self._analyze_trend(hist)
        momentum = self._analyze_momentum(hist)
        macd = self._analyze_macd(hist)
        stoch = self._analyze_stochastic(hist)
        # Licznik sygnałów
        bullish_signals = 0
        bearish_signals = 0
        # Analiza trendu
        if trend == "Trend wzrostowy":
            bullish_signals += 2
        elif trend == "Trend spadkowy":
            bearish_signals += 2
        # Analiza momentum
        if momentum == "Wyprzedany":
            bullish_signals += 1
        elif momentum == "Wykupiony":
            bearish_signals += 1
        # Analiza MACD
        if macd == "Wzrostowy":
            bullish_signals += 1
        elif macd == "Spadkowy":
            bearish_signals += 1
        # Analiza Stochastic
        if stoch == "Wyprzedany":
            bullish_signals += 1
        elif stoch == "Wykupiony":
            bearish_signals += 1
        # --- Nowa logika: uwzględnij predykcję ML ---
        if ml_pred is not None:
            if ml_pred == 0:  # ML przewiduje spadek
                if bearish_signals >= 2:
                    return "Sygnał sprzedaży"
                else:
                    return "Trzymaj"
            elif ml_pred == 1:  # ML przewiduje wzrost
                if bullish_signals >= 4:
                    return "Silny sygnał kupna"
                elif bullish_signals >= 2:
                    return "Sygnał kupna"
                else:
                    return "Trzymaj"
        # --- Stara logika fallback ---
        if bullish_signals >= 4:
            return "Silny sygnał kupna"
        elif bullish_signals >= 2:
            return "Sygnał kupna"
        elif bearish_signals >= 4:
            return "Silny sygnał sprzedaży"
        elif bearish_signals >= 2:
            return "Sygnał sprzedaży"
        else:
            return "Trzymaj"
    
    def _predict_market_movement(self, hist):
        """Predict next day market movement using technical indicators, ML i sentyment newsów"""
        try:
            features = hist[['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'Stoch', 'Stoch_signal', 'news_sentiment']].dropna()
            target = (hist['Close'].shift(-1) > hist['Close']).astype(int).loc[features.index]
            if self.model is None and len(features) > 30:
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                joblib.dump(model, self.model_path)
                self.model = model
                y_pred = model.predict(X_test)
                # --- METRYKI ---
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                metrics = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                }
                return {"prediction": int(model.predict([features.iloc[-1]])[0]), "metrics": metrics}
            elif self.model is not None:
                return {"prediction": int(self.model.predict([features.iloc[-1]])[0])}
            else:
                return {"prediction": None, "info": "Not enough data to train model."}
        except Exception as e:
            return {"prediction": None, "error": str(e)}
    
    def _load_predictor(self):
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception:
                return None
        return None 

    def generate_plot(self, symbol, period='3mo', indicators=None, output_file='plot.png'):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if len(hist) < 20:
            return None  # Za mało danych do wykresu z zaawansowanymi wskaźnikami
        plt.figure(figsize=(12,6))
        plt.plot(hist['Close'], label='Close')
        if indicators:
            if 'SMA_20' in indicators:
                plt.plot(SMAIndicator(close=hist['Close'], window=20).sma_indicator(), label='SMA 20')
            if 'EMA_20' in indicators:
                plt.plot(EMAIndicator(close=hist['Close'], window=20).ema_indicator(), label='EMA 20')
            if 'ATR_14' in indicators:
                plt.plot(AverageTrueRange(high=hist['High'], low=hist['Low'], close=hist['Close'], window=14).average_true_range(), label='ATR 14')
            if 'ADX_14' in indicators:
                plt.plot(ADXIndicator(high=hist['High'], low=hist['Low'], close=hist['Close'], window=14).adx(), label='ADX 14')
            if 'CCI_20' in indicators:
                plt.plot(CCIIndicator(high=hist['High'], low=hist['Low'], close=hist['Close'], window=20).cci(), label='CCI 20')
            if 'OBV' in indicators:
                plt.plot(OnBalanceVolumeIndicator(close=hist['Close'], volume=hist['Volume']).on_balance_volume(), label='OBV')
        plt.title(f'{symbol} Price & Indicators')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        return output_file

    def plot_stock_data(self, symbol, data):
        plt.switch_backend('Agg')  # Ensure we're using the non-interactive backend
        plt.figure(figsize=(12,6))
        plt.plot(data['Close'], label='Close Price')
        plt.title(f'{symbol} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{symbol}_stock_price.png')
        plt.close()

    def format_symbol(self, symbol):
        """Formatuje symbol do odpowiedniego formatu dla yfinance"""
        # Usuń przedrostek NASDAQ: jeśli istnieje
        symbol = symbol.replace('NASDAQ:', '')
        
        # Sprawdź czy to kryptowaluta
        symbol_upper = symbol.upper()
        if symbol_upper in self.crypto_mapping:
            return self.crypto_mapping[symbol_upper]
        elif symbol_upper.endswith('USD'):
            return symbol_upper
        return symbol