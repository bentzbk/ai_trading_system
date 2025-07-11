import schedule
import time
import subprocess
import logging
from datetime import datetime, timedelta
import os
import sys

class TradingScheduler:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        
        # Check if it's a weekend
        if now.weekday() > 4:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's during market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def run_trading_cycle(self):
        """Run trading cycle if market is open"""
        if self.is_market_open():
            self.logger.info("Market is open - running trading cycle")
            try:
                from trading_engine import TradingEngine
                engine = TradingEngine()
                engine.execute_trading_cycle()
            except Exception as e:
                self.logger.error(f"Error in trading cycle: {e}")
        else:
            self.logger.info("Market is closed - skipping trading cycle")
    
    def run_model_training(self):
        """Run model training (daily after market close)"""
        self.logger.info("Starting daily model training")
        try:
            # Run Colab training script
            subprocess.run([
                sys.executable, 
                "colab_training.py"
            ], check=True)
            self.logger.info("Model training completed")
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
    
    def run_data_collection(self):
        """Run data collection and preprocessing"""
        self.logger.info("Starting data collection")
        try:
            from data_collector import DataCollector
            collector = DataCollector()
            symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX',
                'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'SPOT', 'UBER'
            ]
            data = collector.get_market_data(symbols, period='1mo')
            self.logger.info(f"Data collection completed for {len(data)} symbols")
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        self.logger.info("Starting automated scheduler...")
        
        # Schedule trading every 15 minutes during market hours
        schedule.every(15).minutes.do(self.run_trading_cycle)
        
        # Schedule daily model training at 6 PM (after market close)
        schedule.every().day.at("18:00").do(self.run_model_training)
        
        # Schedule data collection every hour during market hours
        schedule.every().hour.do(self.run_data_collection)
        
        self.logger.info("Scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")

if __name__ == "__main__":
    scheduler = TradingScheduler()
    scheduler.start_scheduler()
