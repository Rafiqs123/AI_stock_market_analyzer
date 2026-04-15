# 📊 AI Stock Market Analyzer

Interactive application for forecasting stock and cryptocurrency prices using Machine Learning. Integrates Random Forest, technical analysis (SMA, EMA, MACD, RSI), financial news retrieval, strategy backtesting, and data visualization. Web interface with WebSocket for real-time communication.

## ✨ Key Features

- 🤖 **Price Prediction** - Random Forest model for forecasting price increases/decreases
- 📈 **Technical Analysis** - Technical indicators: SMA, EMA, MACD, RSI, Bollinger Bands, ATR and more
- 💬 **Interactive Chatbot** - WebSocket interface for real-time communication with models
- 📰 **News Integration** - Retrieve and analyze financial news (NewsAPI)
- 📊 **Data Visualization** - Dynamic charts of prices and technical indicators
- 🔄 **Strategy Backtesting** - Simulate trading strategies on historical data
- ⚙️ **Parameter Optimization** - Auto-tune model for better results
- 💾 **Caching** - Redis for caching news and market data
- 📱 **Responsive Interface** - Web application with mobile device support
<img width="1897" height="1057" alt="zdj1" src="https://github.com/user-attachments/assets/c5e5bde7-4457-489f-bab3-5786362acc16" />
<img width="993" height="1027" alt="zdj2" src="https://github.com/user-attachments/assets/04fbf95c-de9d-469b-81ec-c5a7b077bbbe" />
<img width="993" height="998" alt="zdj3" src="https://github.com/user-attachments/assets/e2a968a1-184d-467e-b75f-d6a9192dd90f" />
<img width="993" height="1025" alt="zdj4" src="https://github.com/user-attachments/assets/522404cc-d28f-487c-a4b6-15819762337f" />


## 🛠️ Technology Stack

### Backend
- **Framework:** Flask, Flask-SocketIO
- **Machine Learning:** scikit-learn (Random Forest)
- **Deep Learning:** transformers, torch

### Market Data
- **Data Retrieval:** yfinance, Binance API
- **Technical Analysis:** TA-Lib
- **Financial News:** NewsAPI

### Frontend
- **Languages:** HTML5, CSS, JavaScript
- **Communication:** WebSockets (Socket.IO)

### Infrastructure
- **Caching:** Redis
- **Monitoring:** Sentry, Weights & Biases
- **Testing:** pytest, Flask-Testing

## 📋 Requirements

- Python 3.8+
- Redis (for caching)
- API keys:
  - News API Key (https://newsapi.org)
  - (Optional) Binance API
