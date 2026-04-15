from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from model_trainer import ModelTrainer
from datetime import datetime
import numpy as np  # Dodano do statystyk transakcji

app = Flask(__name__)
socketio = SocketIO(app)
trainer = ModelTrainer()
current_symbol = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(msg):
    global current_symbol
    # Accept both string and dict (for backward compatibility)
    if isinstance(msg, dict):
        text = msg.get('text', '').strip()
        indicators = msg.get('indicators', None)
    else:
        text = msg.strip()
        indicators = None
    text_lower = text.lower()
    # Reset command
    if text_lower == 'reset':
        current_symbol = None
        emit('message', {'response': "Symbol reset. Please enter a new stock symbol."})
    # Plot command
    elif text_lower == 'plot':
        if current_symbol:
            emit('message', {'response': f"📊 Generating price chart for {current_symbol}..."})
            img = trainer.generate_plot(current_symbol)
            if img:
                emit('plot', {'image': img})
                emit('message', {'response': f"📈 Price chart for {current_symbol} generated successfully!\nYou can type another symbol, 'plot' for a new chart, 'retrain' to retrain the model, or 'reset' to change the symbol."})
            else:
                emit('message', {'response': f"❌ Could not generate plot for {current_symbol}. Please try again or check if the symbol is valid."})
        else:
            emit('message', {'response': "No symbol selected. Please enter a stock symbol first."})
    # Help command
    elif text_lower == 'help':
        emit('message', {'response':
            "**Available commands:**\n"
            "- Enter a stock symbol (e.g. NVDA, AAPL, TSLA)\n"
            "- 'plot' — show price chart for current symbol\n"
            "- 'reset' — change the symbol\n"
            "- 'retrain' — retrain the model for the current symbol\n"
            "- 'params' — show current model parameters\n"
            "- 'optimize' — auto-optimize model parameters\n"
            "- 'help' — show this help message\n"
            "- 'clear' — clear chat window\n"
        })
    # Clear command
    elif text_lower == 'clear':
        emit('message', {'response': "Chat cleared."})
    # Params command
    elif text_lower == 'params':
        params = trainer.get_model_parameters()
        emit('message', {'response': 
            f"**Current Model Parameters:**\n"
            f"• max_depth: {params['max_depth']}\n"
            f"• min_samples_leaf: {params['min_samples_leaf']}\n"
            f"• n_estimators: {params['n_estimators']}\n"
            f"• min_samples_split: {params['min_samples_split']}\n\n"
            f"💡 Use 'optimize' to auto-adjust parameters for better performance"
        })
    # Optimize command
    elif text_lower == 'optimize':
        if current_symbol:
            emit('message', {'response': "🔄 Auto-optimizing model parameters..."})
            # Set more conservative parameters
            trainer.set_model_parameters(
                max_depth=5,
                min_samples_leaf=12,
                n_estimators=15,
                min_samples_split=6
            )
            emit('message', {'response': "✅ Model parameters optimized! Use 'retrain' to apply changes."})
        else:
            emit('message', {'response': "No symbol selected. Please enter a stock symbol first."})
    # Retrain command
    elif text_lower == 'retrain':
        if current_symbol:
            emit('message', {'response': f"Retraining model for {current_symbol}..."})
            success = trainer.train(current_symbol)
            if success:
                # BACKTEST: log test metrics to console after retrain
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    X, y = trainer.get_features_and_labels(current_symbol)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    X_train_scaled = trainer.scaler.fit_transform(X_train)
                    X_test_scaled = trainer.scaler.transform(X_test)
                    trainer.model.fit(X_train_scaled, y_train)
                    y_pred = trainer.model.predict(X_test_scaled)
                    test_acc = accuracy_score(y_test, y_pred)
                    test_prec = precision_score(y_test, y_pred, zero_division=0)
                    test_rec = recall_score(y_test, y_pred, zero_division=0)
                    test_f1 = f1_score(y_test, y_pred, zero_division=0)
                    # Test period start and end dates
                    # Use the same robust data fetching as training
                    df = trainer._fetch_data_direct_api(current_symbol)
                    if df is None or len(df) < 30:
                        # Fallback to yfinance
                        import yfinance as yf
                        ticker = yf.Ticker(current_symbol)
                        try:
                            df = ticker.history(period='1y', interval='1d')
                        except:
                            df = ticker.history(period='6mo', interval='1d')
                    test_len = len(y_test)
                    test_start = df.index[-test_len] if test_len <= len(df) else df.index[0]
                    test_end = df.index[-1]
                    print("\n=== BACKTEST RESULTS (20% last samples) ===")
                    print(f"Test period: {test_start.date()} to {test_end.date()} ({test_len} days)")
                    print(f"Test Accuracy:  {test_acc:.2%}")
                    print(f"Test Precision: {test_prec:.2%}")
                    print(f"Test Recall:    {test_rec:.2%}")
                    print(f"Test F1:        {test_f1:.2%}")
                    # STRATEGY SIMULATION
                    close_prices = df['Close'].iloc[-test_len:].values
                    capital = 10000.0
                    position = 0  # 0 = no position, 1 = bought
                    shares = 0
                    trade_results = []  # list of profit/loss for each trade
                    entry_price = None
                    for i in range(len(y_pred)):
                        if y_pred[i]:  # model predicts UP
                            if position == 0:
                                shares = capital / close_prices[i]
                                entry_price = close_prices[i]
                                capital = 0
                                position = 1
                        else:  # model predicts DOWN
                            if position == 1:
                                exit_price = close_prices[i]
                                trade_pl = (exit_price - entry_price) / entry_price * 100  # %
                                trade_results.append(trade_pl)
                                capital = shares * close_prices[i]
                                shares = 0
                                position = 0
                    # At the end, sell if still holding
                    if position == 1:
                        exit_price = close_prices[-1]
                        trade_pl = (exit_price - entry_price) / entry_price * 100
                        trade_results.append(trade_pl)
                        capital = shares * close_prices[-1]
                    profit = capital - 10000.0
                    # Trade statistics
                    losses = [x for x in trade_results if x < 0]
                    gains = [x for x in trade_results if x > 0]
                    avg_loss = np.mean(losses) if losses else 0
                    max_loss = np.min(losses) if losses else 0
                    avg_gain = np.mean(gains) if gains else 0
                    max_gain = np.max(gains) if gains else 0
                    print(f"Simulated trading result (test set): {capital:.2f} PLN (profit: {profit:+.2f} PLN)")
                    print(f"Number of trades: {len(trade_results)}")
                    print(f"Average loss per losing trade: {avg_loss:.2f}% | Max loss: {max_loss:.2f}%")
                    print(f"Average gain per winning trade: {avg_gain:.2f}% | Max gain: {max_gain:.2f}%")
                    print("==========================================\n")
                except Exception as e:
                    print(f"Backtest error: {e}")
                emit('message', {'response': f"Model retrained for {current_symbol}. You can now type 'plot', enter a new symbol, or 'reset'."})
            else:
                emit('message', {'response': f"Retraining failed for {current_symbol}. Try another symbol or 'reset'."})
        else:
            emit('message', {'response': "No symbol selected. Please enter a stock symbol first."})
    # Stock symbol (alphanumeric, up to 6 chars)
    elif len(text) <= 6 and text.isalnum():
        current_symbol = text.upper()
        emit('message', {'response': f"Analyzing symbol {current_symbol}..."})
        # Pass indicators to predict if provided
        result = trainer.predict(current_symbol, indicators=indicators)
        if result:
            last_update = datetime.now().strftime('%Y-%m-%d %H:%M')
            lines = [
                f"🔄 Starting prediction for {current_symbol}",
                f"Prediction: {result['prediction']} (Confidence: {result['probability']:.1%})",
                f"Model Accuracy: {result['accuracy']:.1%}",
                f"Last model update: {last_update}",
                "",
                "Detailed Analysis:"
            ]
            for k, v in result['signals'].items():
                lines.append(f"• {k}: {v}")
            lines.append("\nYou can type 'plot' to see the price chart, 'retrain' to retrain the model, or 'reset' to change the symbol.")
            emit('message', {'response': "\n".join(lines)})
        else:
            emit('message', {'response': f"Symbol not found or no data available for '{current_symbol}'. Try another symbol or 'reset'."})
    else:
        emit('message', {'response': "Unknown command. Enter a stock symbol, 'plot', 'retrain', 'reset', or 'help'."})

if __name__ == '__main__':
    socketio.run(app, debug=True)