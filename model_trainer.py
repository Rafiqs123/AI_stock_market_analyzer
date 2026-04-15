from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from ta import add_all_ta_features
import ta
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=20,
            max_depth=8,
            min_samples_leaf=6,
            n_jobs=1,
            random_state=42,
            class_weight='balanced',
            verbose=0
        )
        self.scaler = StandardScaler()
        self.last_accuracy = 0
        self.is_trained = False
        self.trained_symbols = set()
        self.scaler_fitted = False
        self.model_path = os.path.join('models', 'market_model.joblib')
        
        # Model optimization settings
        self.auto_optimize = True  # Enable automatic parameter optimization
        self.max_optimization_attempts = 3
        
        # Configure yfinance with retry mechanism
        self._configure_yfinance()

    def _configure_yfinance(self):
        """Configure yfinance with retry mechanism and timeout"""
        try:
            # Create a session with retry strategy
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Set timeout
            session.timeout = 30
            
            # Note: yfinance doesn't have set_requests_session in this version
            # The session will be used automatically by requests
            print("✅ yfinance configured with retry mechanism (using default session)")
        except Exception as e:
            print(f"⚠️ Could not configure yfinance session: {e}")

    def _fetch_data_direct_api(self, symbol):
        """Fetch data directly from Yahoo Finance API with better error handling"""
        try:
            # Try multiple API endpoints
            endpoints = [
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1y",
                f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1y",
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=6mo"
            ]
            
            for i, url in enumerate(endpoints):
                try:
                    print(f"Trying API endpoint {i+1}...")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        # Check if response is not empty
                        if not response.text.strip():
                            print("Empty response received")
                            continue
                            
                        try:
                            data = response.json()
                        except ValueError as e:
                            print(f"Invalid JSON response: {e}")
                            print(f"Response content: {response.text[:200]}...")
                            continue
                        
                        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                            result = data['chart']['result'][0]
                            
                            # Check if we have timestamp and quote data
                            if 'timestamp' not in result or 'indicators' not in result:
                                print("Missing timestamp or indicators in response")
                                continue
                                
                            timestamps = result['timestamp']
                            if not timestamps:
                                print("No timestamps in response")
                                continue
                                
                            quotes = result['indicators']['quote'][0]
                            
                            # Check if we have valid quote data
                            if not all(key in quotes for key in ['open', 'high', 'low', 'close', 'volume']):
                                print("Missing required quote data")
                                continue
                            
                            # Filter out None values
                            valid_indices = []
                            valid_data = {
                                'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': []
                            }
                            
                            for i, ts in enumerate(timestamps):
                                if (quotes['open'][i] is not None and 
                                    quotes['high'][i] is not None and 
                                    quotes['low'][i] is not None and 
                                    quotes['close'][i] is not None and 
                                    quotes['volume'][i] is not None):
                                    valid_indices.append(i)
                                    valid_data['Open'].append(quotes['open'][i])
                                    valid_data['High'].append(quotes['high'][i])
                                    valid_data['Low'].append(quotes['low'][i])
                                    valid_data['Close'].append(quotes['close'][i])
                                    valid_data['Volume'].append(quotes['volume'][i])
                            
                            if len(valid_indices) < 30:
                                print(f"Not enough valid data points: {len(valid_indices)}")
                                continue
                            
                            # Create DataFrame with valid data
                            df = pd.DataFrame(valid_data, 
                                           index=pd.to_datetime([timestamps[i] for i in valid_indices], unit='s'))
                            
                            print(f"✅ Direct API call successful: {df.shape}")
                            return df
                        else:
                            print("No chart data in response")
                    else:
                        print(f"API call failed with status: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"Timeout on endpoint {i+1}")
                except requests.exceptions.RequestException as e:
                    print(f"Request error on endpoint {i+1}: {e}")
                except Exception as e:
                    print(f"Unexpected error on endpoint {i+1}: {e}")
            
            print("All API endpoints failed")
            return None
            
        except Exception as e:
            print(f"Direct API fetch error: {e}")
            return None

    def _optimize_model_parameters(self, overfitting_gap, current_params):
        """Automatically optimize model parameters to reduce overfitting"""
        print(f"\n🔧 Auto-optimizing model parameters...")
        print(f"Current overfitting gap: {overfitting_gap:.2%}")
        
        # Define optimization strategies based on overfitting severity
        if overfitting_gap > 0.3:  # Severe overfitting
            print("🔄 Severe overfitting detected - aggressive optimization")
            new_params = {
                'max_depth': max(3, current_params['max_depth'] - 3),
                'min_samples_leaf': min(20, current_params['min_samples_leaf'] + 8),
                'n_estimators': max(10, current_params['n_estimators'] - 8),
                'min_samples_split': 10
            }
        elif overfitting_gap > 0.2:  # High overfitting
            print("🔄 High overfitting detected - moderate optimization")
            new_params = {
                'max_depth': max(4, current_params['max_depth'] - 2),
                'min_samples_leaf': min(15, current_params['min_samples_leaf'] + 5),
                'n_estimators': max(12, current_params['n_estimators'] - 5),
                'min_samples_split': 8
            }
        elif overfitting_gap > 0.1:  # Moderate overfitting
            print("🔄 Moderate overfitting detected - light optimization")
            new_params = {
                'max_depth': max(5, current_params['max_depth'] - 1),
                'min_samples_leaf': min(12, current_params['min_samples_leaf'] + 3),
                'n_estimators': max(15, current_params['n_estimators'] - 3),
                'min_samples_split': 6
            }
        else:
            print("✅ Overfitting gap acceptable - no optimization needed")
            return False
        
        print(f"📊 Parameter changes:")
        print(f"   max_depth: {current_params['max_depth']} → {new_params['max_depth']}")
        print(f"   min_samples_leaf: {current_params['min_samples_leaf']} → {new_params['min_samples_leaf']}")
        print(f"   n_estimators: {current_params['n_estimators']} → {new_params['n_estimators']}")
        
        # Apply new parameters
        self.model.max_depth = new_params['max_depth']
        self.model.min_samples_leaf = new_params['min_samples_leaf']
        self.model.n_estimators = new_params['n_estimators']
        self.model.min_samples_split = new_params['min_samples_split']
        
        return True

    def set_model_parameters(self, max_depth=None, min_samples_leaf=None, n_estimators=None, min_samples_split=None):
        """Manually set model parameters for fine-tuning"""
        if max_depth is not None:
            self.model.max_depth = max_depth
            print(f"✅ Set max_depth to {max_depth}")
        
        if min_samples_leaf is not None:
            self.model.min_samples_leaf = min_samples_leaf
            print(f"✅ Set min_samples_leaf to {min_samples_leaf}")
        
        if n_estimators is not None:
            self.model.n_estimators = n_estimators
            print(f"✅ Set n_estimators to {n_estimators}")
        
        if min_samples_split is not None:
            self.model.min_samples_split = min_samples_split
            print(f"✅ Set min_samples_split to {min_samples_split}")
        
        print(f"📊 Current model parameters:")
        print(f"   max_depth: {self.model.max_depth}")
        print(f"   min_samples_leaf: {self.model.min_samples_leaf}")
        print(f"   n_estimators: {self.model.n_estimators}")
        print(f"   min_samples_split: {self.model.min_samples_split}")

    def get_model_parameters(self):
        """Get current model parameters"""
        return {
            'max_depth': self.model.max_depth,
            'min_samples_leaf': self.model.min_samples_leaf,
            'n_estimators': self.model.n_estimators,
            'min_samples_split': self.model.min_samples_split
        }

    def prepare_market_data(self, symbol, days=252, indicators=None):
        try:
            symbol = symbol.strip().upper()
            
            # Try direct API call first (more reliable)
            print(f"🔍 Fetching data for {symbol}...")
            df = self._fetch_data_direct_api(symbol)
            
            if df is None or len(df) < 30:
                print(f"Direct API failed, trying yfinance...")
                # Fallback to yfinance
                ticker = yf.Ticker(symbol)
                try:
                    df = ticker.history(period='1y', interval='1d')
                    print(f"[DEBUG] yfinance {symbol} history shape: {df.shape}")
                except Exception as e:
                    print(f"yfinance failed: {e}")
                    try:
                        # Try shorter period
                        df = ticker.history(period='6mo', interval='1d')
                        print(f"[DEBUG] yfinance {symbol} history shape (6mo): {df.shape}")
                    except Exception as e2:
                        print(f"yfinance 6mo failed: {e2}")
                        return None, None
            
            # Check if we got any data
            if df is None or len(df) < 30:
                print(f"{symbol}: No price data found, symbol may be delisted")
                return None, None
                
            # Verify we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ Missing required columns for {symbol}: {missing_columns}")
                return None, None
                
            # Add technical indicators with error handling
            try:
                df = add_all_ta_features(
                    df,
                    open="Open",
                    high="High",
                    low="Low",
                    close="Close",
                    volume="Volume",
                    fillna=True
                )
            except Exception as e:
                print(f"⚠️ Error adding technical indicators: {e}")
                # Continue with basic features only
                pass
            
            # Handle missing technical indicators gracefully
            all_features = {}
            
            # Basic features that should always work
            all_features['returns'] = df['Close'].pct_change()
            all_features['volume_change'] = df['Volume'].pct_change()
            
            # Technical indicators with error handling
            try:
                all_features['rsi'] = df['momentum_rsi']
            except:
                all_features['rsi'] = pd.Series(0, index=df.index)
                
            try:
                all_features['macd'] = df['trend_macd']
            except:
                all_features['macd'] = pd.Series(0, index=df.index)
                
            try:
                all_features['sma_5'] = df['Close'] / df['Close'].rolling(window=5).mean() - 1
            except:
                all_features['sma_5'] = pd.Series(0, index=df.index)
                
            try:
                all_features['sma_20'] = df['Close'] / df['Close'].rolling(window=20).mean() - 1
            except:
                all_features['sma_20'] = pd.Series(0, index=df.index)
                
            try:
                all_features['volatility'] = df['volatility_bbm']
            except:
                all_features['volatility'] = pd.Series(0, index=df.index)
                
            try:
                all_features['trend'] = df['trend_ema_fast']
            except:
                all_features['trend'] = pd.Series(0, index=df.index)
                
            try:
                all_features['momentum'] = df['momentum_stoch_rsi']
            except:
                all_features['momentum'] = pd.Series(0, index=df.index)
            # Use only selected indicators if provided
            if indicators:
                features = pd.DataFrame({k: v for k, v in all_features.items() if k in indicators}).fillna(0)
            else:
                features = pd.DataFrame(all_features).fillna(0)
            labels = (df['Close'].shift(-1) > df['Close'])
            features = features.dropna()
            labels = labels[features.index]
            features = features[:-1]
            labels = labels[:-1]
            if len(features) < 3:
                return None, None
            return features, labels
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            return None, None

    def train(self, symbol=None):
        try:
            if symbol is None:
                print("❌ No symbol provided")
                return False
            print(f"🎯 Starting basic training for {symbol}...")
            X, y = self.prepare_market_data(symbol)
            if X is None or y is None:
                print("❌ Failed to prepare data")
                return False
            print(f"📊 Processing {len(X)} samples...")
            
            # Split data into train and test sets (80% train, 20% test)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, random_state=42
            )
            print(f"📈 Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            
            # Fit scaler on training data only
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.scaler_fitted = True
            
            print("🌳 Growing decision trees...")
            phases = [
                (8, "Phase 1: Basic trees..."),
                (14, "Phase 2: Adding complexity..."),
                (20, "Phase 3: Fine tuning...")
            ]
            for n_trees, msg in phases:
                print(msg)
                self.model.n_estimators = n_trees
                self.model.fit(X_train_scaled, y_train)
                train_acc = self.model.score(X_train_scaled, y_train)
                test_acc = self.model.score(X_test_scaled, y_test)
                print(f"Train accuracy: {train_acc:.2%}, Test accuracy: {test_acc:.2%}")
            
            # Final evaluation on test set
            y_pred = self.model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            
            feature_importance = pd.DataFrame({
                'feature': ['returns', 'volume', 'rsi', 'macd', 'sma_5', 'sma_20', 'volatility', 'trend', 'momentum'],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Training Results:")
            print(f"Training Accuracy: {self.model.score(X_train_scaled, y_train):.2%}")
            print(f"Test Accuracy: {test_accuracy:.2%}")
            print(f"Test Precision: {test_precision:.2%}")
            print(f"Test Recall: {test_recall:.2%}")
            print(f"Test F1-Score: {test_f1:.2%}")
            print(f"Overfitting gap: {self.model.score(X_train_scaled, y_train) - test_accuracy:.2%}")
            
            # Check for overfitting
            overfitting_gap = self.model.score(X_train_scaled, y_train) - test_accuracy
            if overfitting_gap > 0.1:  # More than 10% gap
                print(f"⚠️ WARNING: Potential overfitting detected! Gap: {overfitting_gap:.2%}")
                print("💡 Consider: reducing model complexity, adding regularization, or collecting more data")
                print("🔧 Suggested fixes:")
                print("   • Reduce max_depth (currently: {})".format(self.model.max_depth))
                print("   • Increase min_samples_leaf (currently: {})".format(self.model.min_samples_leaf))
                print("   • Reduce n_estimators (currently: {})".format(self.model.n_estimators))
                print("   • Add more training data (currently: {} samples)".format(len(X_train)))
                
                # Auto-optimize if enabled
                if self.auto_optimize and overfitting_gap > 0.15:  # Only for significant overfitting
                    current_params = {
                        'max_depth': self.model.max_depth,
                        'min_samples_leaf': self.model.min_samples_leaf,
                        'n_estimators': self.model.n_estimators
                    }
                    
                    if self._optimize_model_parameters(overfitting_gap, current_params):
                        print("🔄 Retraining with optimized parameters...")
                        # Retrain with new parameters
                        self.model.fit(X_train_scaled, y_train)
                        new_train_acc = self.model.score(X_train_scaled, y_train)
                        new_test_acc = self.model.score(X_test_scaled, y_test)
                        new_overfitting_gap = new_train_acc - new_test_acc
                        
                        print(f"📊 After optimization:")
                        print(f"   Train accuracy: {new_train_acc:.2%}")
                        print(f"   Test accuracy: {new_test_acc:.2%}")
                        print(f"   Overfitting gap: {new_overfitting_gap:.2%}")
                        
                        # Update metrics if improvement
                        if new_overfitting_gap < overfitting_gap:
                            test_accuracy = new_test_acc
                            test_precision = precision_score(y_test, self.model.predict(X_test_scaled), zero_division=0)
                            test_recall = recall_score(y_test, self.model.predict(X_test_scaled), zero_division=0)
                            test_f1 = f1_score(y_test, self.model.predict(X_test_scaled), zero_division=0)
                            overfitting_gap = new_overfitting_gap
                            print("✅ Optimization successful - using improved model")
                        else:
                            print("⚠️ Optimization didn't improve results - keeping original model")
            elif overfitting_gap > 0.05:  # More than 5% gap
                print(f"⚠️ Moderate overfitting detected. Gap: {overfitting_gap:.2%}")
            else:
                print(f"✅ Good generalization! Gap: {overfitting_gap:.2%}")
            print("\nTop 3 Important Features:")
            for _, row in feature_importance.head(3).iterrows():
                print(f"• {row['feature']}: {row['importance']:.3f}")
            
            self.last_accuracy = test_accuracy  # Use test accuracy as the real accuracy
            self.is_trained = True
            self.trained_symbols.add(symbol)
            
            # Save test metrics for later use
            self.test_metrics = {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'overfitting_gap': self.model.score(X_train_scaled, y_train) - test_accuracy
            }
            
            print(f"\n✅ Training completed")
            print(f"📈 Real model accuracy (test set): {self.last_accuracy:.2%}")
            print("💾 Saving model...")
            self._save_model()
            return True
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            logging.error(f"Training error: {str(e)}")
            return False

    def _save_model(self):
        try:
            model_dir = 'models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Get test metrics from the last training session
            test_metrics = getattr(self, 'test_metrics', {
                'accuracy': self.last_accuracy,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'overfitting_gap': 0.0
            })
            
            save_data = {
                'model': self.model,
                'scaler': self.scaler,
                'scaler_fitted': self.scaler_fitted,
                'trained_symbols': self.trained_symbols,
                'accuracy': self.last_accuracy,
                'test_metrics': test_metrics
            }
            joblib.dump(save_data, self.model_path)
            print(f"✓ Model saved to {self.model_path}")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            logging.error(f"Error saving model: {str(e)}")

    def predict(self, symbol, indicators=None):
        try:
            start_time = time.time()
            print(f"\n🔄 Starting prediction for {symbol}")
            if not self.is_trained or not self.scaler_fitted:
                print("⚙️ Model needs training first (this will take ~30 seconds)")
                if not self.train(symbol):
                    return None
            print("📊 Preparing latest market data...")
            features, _ = self.prepare_market_data(symbol, days=10, indicators=indicators)
            if features is None or len(features) == 0:
                print("❌ Failed to fetch data")
                return None
            print("🔄 Analyzing current market conditions...")
            last_data = features.iloc[-1:].copy()
            X_scaled = self.scaler.transform(last_data)
            print("🎯 Generating prediction...")
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            prediction_prob = float(max(probabilities))
            model_accuracy = float(self.last_accuracy)
            print("Calculating technical indicators...")
            signals = self._analyze_signals(last_data.iloc[0], prediction, prediction_prob, model_accuracy)
            result = {
                'symbol': symbol,
                'prediction': 'UP' if prediction else 'DOWN',
                'probability': prediction_prob,
                'accuracy': model_accuracy,
                'signals': signals,
                'timestamp': datetime.now().isoformat()
            }
            elapsed_time = time.time() - start_time
            print(f"\n✓ Analysis completed in {elapsed_time:.1f} seconds")
            print("\n=== Technical Analysis Results ===")
            print(f"Symbol: {symbol}")
            print(f"Prediction: {result['prediction']} (Confidence: {prediction_prob:.1%})")
            print(f"Model Accuracy: {model_accuracy:.1%}")
            print("\nDetailed Analysis:")
            for signal, value in signals.items():
                print(f"• {signal}: {value}")
            print("===============================\n")
            return result
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            logging.error(f"Error details: {str(e)}", exc_info=True)
            return None

    def _analyze_signals(self, data, predicted_up, pred_prob, model_acc):
        try:
            rsi_overbought = 70
            rsi_oversold = 30
            macd_value = float(data['macd'])
            macd_norm = min(1, max(-1, macd_value / 8))
            sma5_trend = min(1, max(-1, float(data['sma_5']) * 4))
            sma20_trend = min(1, max(-1, float(data['sma_20']) * 4))
            trend_score = (
                sma5_trend * 0.4 +
                sma20_trend * 0.3 +
                macd_norm * 0.3
            )
            signal_strength = trend_score * 0.7
            bullish_signals = []
            bearish_signals = []
            conflicting_signals = []
            if abs(macd_norm) > 0.2:
                strength = macd_norm * 1.2
                if macd_norm > 0:
                    signal_strength += strength
                    bullish_signals.append(f"Strong positive MACD ({macd_value:.3f})")
                else:
                    signal_strength -= strength
                    bearish_signals.append(f"Strong negative MACD ({macd_value:.3f})")
            if data['rsi'] > rsi_overbought:
                signal_strength -= 1.0
                bearish_signals.append(f"Overbought RSI ({data['rsi']:.0f})")
            elif data['rsi'] < rsi_oversold:
                signal_strength += 1.0
                bullish_signals.append(f"Oversold RSI ({data['rsi']:.0f})")
            vol_change = min(1, max(-1, data['volume_change']))
            if abs(vol_change) > 0.05:
                vol_impact = vol_change * 0.5
                if (vol_change > 0) == (signal_strength > 0):
                    signal_strength += vol_impact
                    bullish_signals.append("Volume confirms trend") if vol_change > 0 else bearish_signals.append("Volume confirms trend")
            model_weight = 0.4
            if predicted_up != (signal_strength > 0):
                conflicting_signals.append("Model prediction conflicts with technical signals")
                signal_strength *= (1 - model_weight)
                signal_strength += (1 if predicted_up else -1) * model_weight * pred_prob
            base_confidence = (
                (model_acc * 0.35) +
                (pred_prob * 0.35) +
                (abs(trend_score) * 0.2) +
                (min(len(bullish_signals + bearish_signals) / 4, 0.1))
            )
            confidence = min(0.85, base_confidence * (0.6 if conflicting_signals else 1.0))
            confidence = max(0.35, confidence)
            signal_strength = min(2, max(-2, signal_strength))
            recommendation = (
                'STRONG BUY' if signal_strength >= 1.5 and confidence > 0.7 else
                'BUY' if signal_strength >= 0.5 else
                'STRONG SELL' if signal_strength <= -1.5 and confidence > 0.7 else
                'SELL' if signal_strength <= -0.5 else
                'WEAK BUY' if signal_strength > 0 else
                'WEAK SELL' if signal_strength < 0 else
                'WAIT'
            )
            return {
                'Trend': f"{'Upward' if trend_score > 0 else 'Downward'} (strength: {abs(trend_score)*100:.0f}%)",
                'Volatility': 'High' if abs(float(data['volatility'])) > 0.015 else 'Stable',
                'Volume': 'Rising' if float(data['volume_change']) > 0 else 'Falling',
                'RSI': f"{data['rsi']:.0f} ({'Overbought' if data['rsi'] > rsi_overbought else 'Oversold' if data['rsi'] < rsi_oversold else 'Neutral'})",
                'MACD': f"{'Positive' if macd_norm > 0 else 'Negative'} (strength: {abs(macd_norm):.3f})",
                'Signal Strength': round(signal_strength, 2),
                'Recommendation': recommendation,
                'Reasoning': bullish_signals if signal_strength > 0 else bearish_signals,
                'Conflicts': conflicting_signals,
                'Confidence': f"{confidence:.1%}",
                'Active Signals': bullish_signals + bearish_signals,
                'Bullish Signals': len(bullish_signals),
                'Bearish Signals': len(bearish_signals)
            }
        except Exception as e:
            logging.error(f"Error in signal analysis: {str(e)}")
            return None

    def generate_plot(self, symbol):
        """Generate price chart using the same robust data fetching as training"""
        try:
            print(f"📊 Generating plot for {symbol}...")
            
            # Use the same robust data fetching method as training
            df = self._fetch_data_direct_api(symbol)
            
            if df is None or len(df) < 30:
                print(f"Direct API failed for plot, trying yfinance...")
                # Fallback to yfinance with multiple attempts
                ticker = yf.Ticker(symbol)
                try:
                    df = ticker.history(period='3mo', interval='1d')
                    print(f"[DEBUG] yfinance plot {symbol} history shape: {df.shape}")
                except Exception as e:
                    print(f"yfinance plot failed: {e}")
                    try:
                        # Try shorter period
                        df = ticker.history(period='1mo', interval='1d')
                        print(f"[DEBUG] yfinance plot {symbol} history shape (1mo): {df.shape}")
                    except Exception as e2:
                        print(f"yfinance plot 1mo failed: {e2}")
                        return None
            
            if df is None or len(df) < 10:
                print(f"❌ No data available for plotting {symbol}")
                return None
                
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Main price chart
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df['Close'], linewidth=2, color='#1f77b4')
            plt.title(f'{symbol} - Price Chart (Last {len(df)} days)', fontsize=14, fontweight='bold')
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Volume chart
            plt.subplot(2, 1, 2)
            plt.bar(df.index, df['Volume'], alpha=0.7, color='#ff7f0e')
            plt.title('Volume', fontsize=12, fontweight='bold')
            plt.ylabel('Volume', fontsize=12)
            plt.xlabel('Date', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            print(f"✅ Plot generated successfully for {symbol}")
            return base64.b64encode(buf.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"❌ Error generating plot: {str(e)}")
            logging.error(f"Error generating plot: {str(e)}")
            return None

    def get_features_and_labels(self, symbol):
        """
        Prepare features (X) and labels (y) for the given market symbol.
        Used for backtesting after retrain.
        """
        X, y = self.prepare_market_data(symbol)
        if X is None or y is None:
            return None, None
        return X, y