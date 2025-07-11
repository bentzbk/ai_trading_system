# Run this in Google Colab
!pip install -q torch transformers yfinance alpha-vantage ta scikit-learn python-dotenv huggingface-hub

import os
import torch
from google.colab import drive
from huggingface_hub import HfApi, upload_file
import zipfile

# Mount Google Drive
drive.mount('/content/drive')

# Set up environment variables
os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_alpha_vantage_key'
os.environ['NEWS_API_KEY'] = 'your_news_api_key'
os.environ['HUGGINGFACE_TOKEN'] = 'your_hf_token'

# Copy your code files to Colab
# (Upload data_collector.py and model.py to your Colab environment)

from data_collector import DataCollector
from model import TradingModelTrainer

def automated_training():
    """Automated training function for Colab"""
    print("Starting automated training...")
    
    # Define symbols to trade
    symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SPOT', 'UBER'
    ]
    
    # Collect fresh data
    print("Collecting market data...")
    collector = DataCollector()
    market_data = collector.get_market_data(symbols, period='2y')
    
    # Train model
    print("Training model...")
    trainer = TradingModelTrainer(model_name="trading_model_v1")
    model = trainer.train_model(market_data, epochs=100, batch_size=64)
    
    # Save to Drive
    print("Saving model to Drive...")
    os.makedirs('/content/drive/MyDrive/trading_models', exist_ok=True)
    
    # Copy model files
    import shutil
    shutil.copy('trading_model_v1.pth', '/content/drive/MyDrive/trading_models/')
    
    # Upload to Hugging Face
    print("Uploading to Hugging Face...")
    api = HfApi()
    
    try:
        upload_file(
            path_or_fileobj="trading_model_v1.pth",
            path_in_repo="trading_model_v1.pth",
            repo_id="bentzbk/woof_trade_organziation",
            token=os.environ['HUGGINGFACE_TOKEN']
        )
        print("Model uploaded successfully!")
    except Exception as e:
        print(f"Upload failed: {e}")
    
    print("Training completed!")

# Schedule automated training
import schedule
import time

def schedule_training():
    """Schedule training runs"""
    # Train every day at 6 PM EST (after market close)
    schedule.every().day.at("18:00").do(automated_training)
    
    print("Training scheduled. Running scheduler...")
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour

# Run training immediately
automated_training()

# Uncomment to run scheduled training
# schedule_training()
