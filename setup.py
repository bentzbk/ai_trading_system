#!/usr/bin/env python3
"""
Complete setup script for AI Trading System
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    requirements = [
        "torch==2.0.1",
        "transformers==4.30.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "requests==2.31.0",
        "yfinance==0.2.18",
        "alpha-vantage==2.3.1",
        "scikit-learn==1.3.0",
        "ta==0.10.2",
        "schedule==1.2.0",
        "python-dotenv==1.0.0",
        "huggingface-hub==0.16.4",
        "datasets==2.14.0",
        "accelerate==0.20.3",
        "gradio==3.35.2"
    ]
    
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed successfully!")

def create_env_file():
    """Create environment file template"""
    env_template = """# AI Trading System Environment Variables
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
NEWS_API_KEY=your_news_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
TRADERSPOST_WEBHOOK=https://webhooks.traderspost.io/trading/webhook/a81d6a0c-d3c7-4658-8e46-eb5bb8ae0003/d62cdecb7d7dfeb6d898e296d968f406

# Optional: Add other API keys
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
"""
    
    with open('.env', 'w') as f:
        f.write(env_template)
    
    print("✅ Environment file created (.env)")
    print("🔧 Please edit .env file with your actual API keys")

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'data',
        'logs',
        'backtest_results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created successfully!")

def run_initial_setup():
    """Run initial setup and testing"""
    print("Running initial setup...")
    
    try:
        # Test data collection
        print("Testing data collection...")
        from data_collector import DataCollector
        collector = DataCollector()
        test_data = collector.get_stock_data('AAPL', period='1mo')
        
        if test_data is not None:
            print("✅ Data collection test passed!")
        else:
            print("❌ Data collection test failed - check your API keys")
            return False
        
        # Test model training (small test)
        print("Testing model training...")
        from model import TradingModelTrainer
        trainer = TradingModelTrainer()
        
        # Create small test dataset
        test_market_data = {'AAPL': test_data}
        model = trainer.train_model(test_market_data, epochs=5)
        
        if model is not None:
            print("✅ Model training test passed!")
        else:
            print("❌ Model training test failed")
            return False
        
        print("✅ Initial setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Trading System Setup")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Create environment file
    create_env_file()
    
    # Create directories
    create_directories()
    
    # Ask user to configure API keys
    print("\n⚠️  IMPORTANT: Please configure your API keys in the .env file before proceeding")
    print("Required API keys:")
    print("1. Alpha Vantage API Key (free at https://www.alphavantage.co/support/#api-key)")
    print("2. News API Key (free at https://newsapi.org/)")
    print("3. Hugging Face Token (free at https://huggingface.co/settings/tokens)")
    
    response = input("\nHave you configured your API keys? (y/n): ")
    
    if response.lower() == 'y':
        # Run initial setup
        if run_initial_setup():
            print("\n🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Run 'python trading_engine.py' to test the system")
            print("2. Run 'python scheduler.py' to start automated trading")
            print("3. Monitor logs in the 'logs' directory")
            print("\n⚠️  WARNING: Start with paper trading and small amounts!")
        else:
            print("\n❌ Setup failed. Please check your API keys and try again.")
    else:
        print("\n📝 Please configure your API keys in the .env file and run this script again.")

if __name__ == "__main__":
    main()
