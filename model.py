import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import os

class TradingTransformer(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=128, num_layers=3, num_heads=8, dropout=0.1):
        super(TradingTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # buy, sell, hold
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.position_size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use the last token for prediction
        x = x[:, -1, :]  # (batch_size, hidden_dim)
        
        # Get predictions
        action_logits = self.classifier(x)
        confidence = self.confidence_head(x)
        position_size = self.position_size_head(x)
        
        return action_logits, confidence, position_size

class TradingModelTrainer:
    def __init__(self, model_name="trading_model"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def prepare_features(self, market_data):
        """Prepare features for training"""
        all_features = []
        all_labels = []
        
        for symbol, df in market_data.items():
            # Remove rows with NaN values
            df = df.dropna()
            
            if len(df) < 60:  # Need at least 60 days of data
                continue
            
            # Feature columns
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
                'volume_sma', 'vwap', 'ema_12', 'ema_26', 'sma_20', 'sma_50',
                'atr', 'stoch', 'williams_r', 'avg_sentiment', 'sentiment_std'
            ]
            
            # Select available features
            available_features = [col for col in feature_cols if col in df.columns]
            feature_data = df[available_features].values
            
            # Create sequences
            sequence_length = 30
            for i in range(sequence_length, len(feature_data)):
                # Features: last 30 days
                features = feature_data[i-sequence_length:i]
                
                # Label: next day's price movement
                current_price = df.iloc[i-1]['Close']
                next_price = df.iloc[i]['Close']
                price_change = (next_price - current_price) / current_price
                
                # Create action label
                if price_change > 0.02:  # 2% gain
                    action = 0  # buy
                elif price_change < -0.02:  # 2% loss
                    action = 1  # sell
                else:
                    action = 2  # hold
                
                all_features.append(features)
                all_labels.append(action)
        
        return np.array(all_features), np.array(all_labels)
    
    def train_model(self, market_data, epochs=100, batch_size=32):
        """Train the trading model"""
        print("Preparing features...")
        features, labels = self.prepare_features(market_data)
        
        if len(features) == 0:
            print("No training data available!")
            return None
        
        print(f"Training data shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Normalize features
        original_shape = features.shape
        features_reshaped = features.reshape(-1, features.shape[-1])
        features_normalized = self.scaler.fit_transform(features_reshaped)
        features = features_normalized.reshape(original_shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)
        
        # Initialize model
        input_dim = features.shape[-1]
        model = TradingTransformer(input_dim=input_dim).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                
                action_logits, confidence, position_size = model(batch_X)
                loss = criterion(action_logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X_train):.4f}")
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            action_logits, confidence, position_size = model(X_test)
            predictions = torch.argmax(action_logits, dim=1)
            accuracy = (predictions == y_test).float().mean()
            print(f"Test Accuracy: {accuracy:.4f}")
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': self.scaler,
            'input_dim': input_dim
        }, f"{self.model_name}.pth")
        
        return model
    
    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = TradingTransformer(input_dim=checkpoint['input_dim']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
        return model
    
    def predict(self, model, features):
        """Make predictions"""
        model.eval()
        
        # Normalize features
        original_shape = features.shape
        features_reshaped = features.reshape(-1, features.shape[-1])
        features_normalized = self.scaler.transform(features_reshaped)
        features = features_normalized.reshape(original_shape)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            action_logits, confidence, position_size = model(features_tensor)
            action_probs = torch.softmax(action_logits, dim=1)
            predicted_action = torch.argmax(action_probs, dim=1)
            
            return {
                'action': predicted_action.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'position_size': position_size.cpu().numpy(),
                'probabilities': action_probs.cpu().numpy()
            }

# Test the model
if __name__ == "__main__":
    from data_collector import DataCollector
    
    # Collect data
    collector = DataCollector()
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    market_data = collector.get_market_data(symbols, period='1y')
    
    # Train model
    trainer = TradingModelTrainer()
    model = trainer.train_model(market_data, epochs=50)
    
    print("Model training completed!")
