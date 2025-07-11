#!/usr/bin/env python3
"""
Quick start script for AI Trading System
"""

import subprocess
import sys
import os

def quick_start():
    """Quick start the trading system"""
    print("🚀 AI Trading System Quick Start")
    print("=" * 40)
    
    # Check if setup is complete
    if not os.path.exists('.env'):
        print("❌ Setup not complete. Please run setup.py first.")
        return
    
    print("Select an option:")
    print("1. Test single stock analysis")
    print("2. Run full trading cycle (test mode)")
    print("3. Start live trading (15-minute intervals)")
    print("4. Train model with fresh data")
    print("5. Run backtest")
    print("6. Deploy to Hugging Face")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == '1':
        test_single_analysis()
    elif choice == '2':
        run_test_cycle()
    elif choice == '3':
        start_live_trading()
    elif choice == '4':
        train_model()
    elif choice == '5':
        run_backtest()
    elif choice == '6':
        deploy_to_huggingface()
    else:
        print("Invalid choice. Please try again.")

def test_single_analysis():
    """Test single stock analysis"""
    print("\n🔍 Testing single stock analysis...")
    
    symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
    
    try:
        from trading_engine import TradingEngine
        engine = TradingEngine()
        
        if not engine.load_model():
            print("❌ Model not found. Please train the model first.")
            return
        
        signal = engine.analyze_symbol(symbol)
        
        if signal:
            print(f"\n📊 Analysis Results for {symbol}:")
            print(f"Action: {signal['action']}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print(f"Position Size: {signal['position_size']:.2%}")
            print(f"Current Price: ${signal['current_price']:.2f}")
            print(f"Volatility: {signal['volatility']:.2%}")
            print(f"Sentiment: {signal['sentiment']:.2f}")
        else:
            print(f"❌ Could not analyze {symbol}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def run_test_cycle():
    """Run full trading cycle in test mode"""
    print("\n🧪 Running full trading cycle (test mode)...")
    
    try:
        from trading_engine import TradingEngine
        engine = TradingEngine()
        
        # Override webhook URL for testing
        engine.webhook_url = "https://httpbin.org/post"  # Test endpoint
        
        engine.execute_trading_cycle()
        print("✅ Test cycle completed successfully!")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def start_live_trading():
    """Start live trading"""
    print("\n🔴 LIVE TRADING MODE")
    print("⚠️  WARNING: This will place real trades!")
    
    confirm = input("Are you sure you want to start live trading? (type 'YES' to confirm): ")
    
    if confirm == 'YES':
        print("🚀 Starting live trading...")
        try:
            from scheduler import TradingScheduler
            scheduler = TradingScheduler()
            scheduler.start_scheduler()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Live trading cancelled.")

def train_model():
    """Train model with fresh data"""
    print("\n🎓 Training model with fresh data...")
    
    try:
        from data_collector import DataCollector
        from model import TradingModelTrainer
        
        # Collect data
        collector = DataCollector()
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META']
        market_data = collector.get_market_data(symbols, period='1y')
        
        # Train model
        trainer = TradingModelTrainer()
        model = trainer.train_model(market_data, epochs=100)
        
        if model:
            print("✅ Model training completed successfully!")
        else:
            print("❌ Model training failed.")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def run_backtest():
    """Run backtest"""
    print("\n📈 Running backtest...")
    
    try:
        from trading_engine import TradingEngine
        engine = TradingEngine()
        
        if not engine.load_model():
            print("❌ Model not found. Please train the model first.")
            return
        
        results = engine.backtest_strategy('2023-01-01', '2023-12-31')
        
        if results:
            print(f"✅ Backtest completed!")
            print(f"Total trades: {results['total_trades']}")
            print(f"Buy trades: {results['buy_trades']}")
            print(f"Sell trades: {results['sell_trades']}")
        else:
            print("❌ Backtest failed.")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def deploy_to_huggingface():
    """Deploy to Hugging Face"""
    print("\n🤗 Deploying to Hugging Face...")
    
    try:
        from huggingface_deploy import HuggingFaceDeployer
        
        repo_name = input("Enter your HuggingFace repository name (username/repo-name): ")
        
        deployer = HuggingFaceDeployer(repo_name)
        deployer.deploy_to_huggingface("trading_model_v1.pth")
        
        print("✅ Deployment completed!")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_start()
