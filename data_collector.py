import yfinance as yf
import requests
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import ta
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

load_dotenv()

class DataCollector:
    def __init__(self):
        self.av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.news_key = os.getenv('NEWS_API_KEY')
        self.ts = TimeSeries(key=self.av_key, output_format='pandas')
        
    def get_stock_data(self, symbol, period='1y'):
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def get_technical_indicators(self, df):
        """Calculate technical indicators"""
        indicators = {}
        
        # Price-based indicators
        indicators['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        indicators['macd'] = ta.trend.MACD(df['Close']).macd()
        indicators['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        indicators['bb_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        indicators['bb_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        indicators['bb_middle'] = ta.volatility.BollingerBands(df['Close']).bollinger_mavg()
        
        # Volume indicators
        indicators['volume_sma'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
        indicators['vwap'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
        
        # Trend indicators
        indicators['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        indicators['ema_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        indicators['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        indicators['sma_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        
        # Volatility indicators
        indicators['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Momentum indicators
        indicators['stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        indicators['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        return pd.DataFrame(indicators)
    
    def get_news_sentiment(self, symbol, days=7):
        """Get news sentiment for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} stock",
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.news_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params)
            news_data = response.json()
            
            if 'articles' in news_data:
                articles = news_data['articles']
                sentiment_scores = []
                
                for article in articles[:20]:  # Limit to 20 articles
                    title = article.get('title', '')
                    description = article.get('description', '')
                    text = f"{title} {description}"
                    
                    # Simple sentiment scoring (you can replace with better model)
                    sentiment = self.simple_sentiment_score(text)
                    sentiment_scores.append(sentiment)
                
                return {
                    'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
                    'sentiment_std': np.std(sentiment_scores) if sentiment_scores else 0,
                    'news_count': len(articles)
                }
            
            return {'avg_sentiment': 0, 'sentiment_std': 0, 'news_count': 0}
        
        except Exception as e:
            print(f"Error getting news sentiment: {e}")
            return {'avg_sentiment': 0, 'sentiment_std': 0, 'news_count': 0}
    
    def simple_sentiment_score(self, text):
        """Simple sentiment scoring (replace with better model)"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'gain', 'up', 'rise', 'bull']
        negative_words = ['bad', 'poor', 'negative', 'loss', 'decline', 'down', 'fall', 'bear', 'crash', 'drop']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def get_market_data(self, symbols, period='1y'):
        """Get complete market data for multiple symbols"""
        market_data = {}
        
        for symbol in symbols:
            print(f"Collecting data for {symbol}...")
            
            # Get stock data
            stock_data = self.get_stock_data(symbol, period)
            if stock_data is None:
                continue
            
            # Get technical indicators
            technical_data = self.get_technical_indicators(stock_data)
            
            # Get news sentiment
            news_sentiment = self.get_news_sentiment(symbol)
            
            # Combine all data
            combined_data = pd.concat([stock_data, technical_data], axis=1)
            
            # Add sentiment data
            for key, value in news_sentiment.items():
                combined_data[key] = value
            
            market_data[symbol] = combined_data
        
        return market_data

# Test the data collector
if __name__ == "__main__":
    collector = DataCollector()
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    data = collector.get_market_data(symbols, period='6mo')
    
    for symbol, df in data.items():
        print(f"\n{symbol} data shape: {df.shape}")
        print(f"Latest data:\n{df.tail(1)}")
