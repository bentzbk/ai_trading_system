import requests
import json
import time
import schedule
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from data_collector import DataCollector
from model import TradingModelTrainer
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class TradingEngine:
    def __init__(self):
        self.webhook_url = os.getenv('TRADERSPOST_WEBHOOK')
        self.collector = DataCollector()
        self.trainer = TradingModelTrainer()
        self.model = None
        self.positions = {}
        self.max_positions = 10
        self.max_position_size = 0.05  # 5% max per position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_engine.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path="trading_model_v1.pth"):
        """Load the trained model"""
        try:
            self.model = self.trainer.load_model(model_path)
            self.logger.info("Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_current_positions(self):
        """Get current positions (implement based on your broker API)"""
        # This is a placeholder - replace with actual broker API call
        return self.positions
    
    def calculate_position_size(self, confidence, volatility, portfolio_value=10000):
        """Calculate position size based on confidence and volatility"""
        base_size = self.max_position_size
        volatility_adjusted = base_size * (1 - min(volatility, 0.5))
        confidence_adjusted = volatility_adjusted * confidence
        
        # Ensure minimum and maximum bounds
        position_size = max(0.01, min(confidence_adjusted, self.max_position_size))
        return position_size
    
    def send_webhook(self, action, symbol, confidence, position_size_pct, current_price):
        """Send trading signal to TradersPost webhook"""
        try:
            # Calculate position size in shares
            portfolio_value = 10000  # Adjust based on your account
            position_value = portfolio_value * position_size_pct
            quantity = int(position_value / current_price)
            
            # Skip if quantity is too small
            if quantity < 1:
                self.logger.info(f"Skipping {symbol} - quantity too small: {quantity}")
                return False
            
            # Create webhook payload
            payload = {
                "timestamp": datetime.now().isoformat(),
                "action": action.lower(),
                "ticker": symbol,
                "quantity": quantity,
                "type": "market",
                "confidence": float(confidence),
                "metadata": {
                    "position_size_pct": float(position_size_pct),
                    "current_price": float(current_price),
                    "stop_loss": float(current_price * (1 - self.stop_loss_pct)) if action == "buy" else float(current_price * (1 + self.stop_loss_pct)),
                    "take_profit": float(current_price * (1 + self.take_profit_pct)) if action == "buy" else float(current_price * (1 - self.take_profit_pct))
                }
            }
            
            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully sent {action} signal for {symbol}")
                return True
            else:
                self.logger.error(f"Webhook failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending webhook: {e}")
            return False
    
    def analyze_symbol(self, symbol):
        """Analyze a single symbol and return trading signal"""
        try:
            # Get recent data
            stock_data = self.collector.get_stock_data(symbol, period='3mo')
            if stock_data is None or len(stock_data) < 30:
                return None
            
            # Get technical indicators
            technical_data = self.collector.get_technical_indicators(stock_data)
            
            # Get news sentiment
            news_sentiment = self.collector.get_news_sentiment(symbol)
            
            # Combine data
            combined_data = pd.concat([stock_data, technical_data], axis=1)
            for key, value in news_sentiment.items():
                combined_data[key] = value
            
            # Clean data
            combined_data = combined_data.dropna()
            
            if len(combined_data) < 30:
                return None
            
            # Prepare features for model
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
                'volume_sma', 'vwap', 'ema_12', 'ema_26', 'sma_20', 'sma_50',
                'atr', 'stoch', 'williams_r', 'avg_sentiment', 'sentiment_std'
            ]
            
            available_features = [col for col in feature_cols if col in combined_data.columns]
            feature_data = combined_data[available_features].tail(30).values
            
            # Reshape for model input
            features = feature_data.reshape(1, 30, len(available_features))
            
            # Make prediction
            prediction = self.trainer.predict(self.model, features)
            
            action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            action = action_map[prediction['action'][0]]
            confidence = prediction['confidence'][0][0]
            position_size = prediction['position_size'][0][0]
            
            # Calculate volatility
            returns = combined_data['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Adjust position size based on volatility
            adjusted_position_size = self.calculate_position_size(
                confidence, volatility
            )
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'position_size': adjusted_position_size,
                'current_price': combined_data['Close'].iloc[-1],
                'volatility': volatility,
                'sentiment': news_sentiment['avg_sentiment']
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def risk_management_filter(self, signals):
        """Apply risk management filters to signals"""
        filtered_signals = []
        current_positions = self.get_current_positions()
        
        for signal in signals:
            # Skip if confidence is too low
            if signal['confidence'] < 0.6:
                continue
            
            # Skip if already at max positions
            if len(current_positions) >= self.max_positions and signal['action'] == 'BUY':
                continue
            
            # Skip if volatility is too high
            if signal['volatility'] > 0.05:  # 5% daily volatility
                continue
            
            # Skip if sentiment is too negative for buy signals
            if signal['action'] == 'BUY' and signal['sentiment'] < -0.3:
                continue
            
            # Skip if sentiment is too positive for sell signals
            if signal['action'] == 'SELL' and signal['sentiment'] > 0.3:
                continue
            
            filtered_signals.append(signal)
        
        # Sort by confidence and take top signals
        filtered_signals.sort(key=lambda x: x['confidence'], reverse=True)
        return filtered_signals[:5]  # Max 5 signals per run
    
    def execute_trading_cycle(self):
        """Execute a complete trading cycle"""
        try:
            self.logger.info("Starting trading cycle...")
            
            # Check if model is loaded
            if self.model is None:
                if not self.load_model():
                    self.logger.error("Cannot execute trading cycle - model not loaded")
                    return
            
            # Define symbols to analyze
            symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX',
                'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SPOT', 'UBER',
                'SPY', 'QQQ', 'IWM', 'VIX'
            ]
            
            # Analyze all symbols
            signals = []
            for symbol in symbols:
                self.logger.info(f"Analyzing {symbol}...")
                signal = self.analyze_symbol(symbol)
                if signal and signal['action'] != 'HOLD':
                    signals.append(signal)
            
            # Apply risk management
            filtered_signals = self.risk_management_filter(signals)
            
            # Execute trades
            for signal in filtered_signals:
                self.logger.info(f"Processing signal: {signal}")
                
                success = self.send_webhook(
                    signal['action'],
                    signal['symbol'],
                    signal['confidence'],
                    signal['position_size'],
                    signal['current_price']
                )
                
                if success:
                    # Update positions tracking
                    if signal['action'] == 'BUY':
                        self.positions[signal['symbol']] = {
                            'entry_price': signal['current_price'],
                            'quantity': int(10000 * signal['position_size'] / signal['current_price']),
                            'timestamp': datetime.now()
                        }
                    elif signal['action'] == 'SELL' and signal['symbol'] in self.positions:
                        del self.positions[signal['symbol']]
                
                time.sleep(2)  # Rate limiting
            
            self.logger.info(f"Trading cycle completed. Processed {len(filtered_signals)} signals.")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def start_trading_bot(self):
        """Start the automated trading bot"""
        self.logger.info("Starting trading bot...")
        
        # Load model
        if not self.load_model():
            self.logger.error("Failed to load model. Exiting.")
            return
        
        # Schedule trading every 15 minutes during market hours
        schedule.every(15).minutes.do(self.execute_trading_cycle)
        
        # Also schedule daily model retraining (optional)
        schedule.every().day.at("18:00").do(self.retrain_model)
        
        self.logger.info("Trading bot started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("Trading bot stopped by user")
    
    def retrain_model(self):
        """Retrain model with fresh data"""
        try:
            self.logger.info("Starting model retraining...")
            
            # Collect fresh data
            symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX',
                'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SPOT', 'UBER'
            ]
            
            market_data = self.collector.get_market_data(symbols, period='1y')
            
            # Retrain model
            self.model = self.trainer.train_model(market_data, epochs=50)
            
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Error during model retraining: {e}")
    
    def backtest_strategy(self, start_date, end_date, initial_capital=10000):
        """Backtest the trading strategy"""
        try:
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # This is a simplified backtest - implement more sophisticated version
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
            portfolio_value = initial_capital
            trades = []
            
            for symbol in symbols:
                # Get historical data
                stock_data = self.collector.get_stock_data(symbol, period='1y')
                if stock_data is None:
                    continue
                
                # Simulate trading signals
                for i in range(30, len(stock_data)):
                    # Get features for this date
                    historical_data = stock_data.iloc[i-30:i]
                    
                    # Simulate analysis (simplified)
                    signal = self.analyze_symbol(symbol)
                    if signal and signal['action'] != 'HOLD':
                        trades.append({
                            'date': stock_data.index[i],
                            'symbol': symbol,
                            'action': signal['action'],
                            'price': stock_data.iloc[i]['Close'],
                            'confidence': signal['confidence']
                        })
            
            # Calculate performance metrics
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            
            self.logger.info(f"Backtest completed: {len(buy_trades)} buy signals, {len(sell_trades)} sell signals")
            
            return {
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'trades': trades
            }
            
        except Exception as e:
            self.logger.error(f"Error during backtesting: {e}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Initialize trading engine
    engine = TradingEngine()
    
    # Test single analysis
    print("Testing single symbol analysis...")
    signal = engine.analyze_symbol('AAPL')
    if signal:
        print(f"Signal for AAPL: {signal}")
    
    # Test full trading cycle (comment out for production)
    print("\nTesting full trading cycle...")
    engine.execute_trading_cycle()
    
    # Uncomment to start live trading
    # engine.start_trading_bot()
