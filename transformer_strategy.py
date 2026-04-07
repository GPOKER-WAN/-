import numpy as np
import pandas as pd
from .core import DepthStrategy
from models.transformer_model import TransformerModel
from sklearn.preprocessing import StandardScaler

class TransformerStrategy(DepthStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.transformer_model = TransformerModel(config)
        self.scaler = StandardScaler()
        self.model_trained = False
    
    def prepare_features(self, data):
        """准备Transformer模型的特征"""
        # 首先使用父类的方法计算技术指标
        data = self.calculate_indicators(data)
        
        # 选择用于模型的特征
        feature_columns = [
            'close', 'volume', 'ATR', 'MA20', 'MA60', 'MA120',
            'BB_Middle', 'BB_Std', 'RSI', 'MACD', 'Signal', 'Histogram',
            'Trend_Strength', 'Volume_Ratio', 'Price_Momentum', 'Price_Volatility',
            'K', 'D', 'ADX'
        ]
        
        # 提取特征
        features = data[feature_columns].copy()
        
        # 处理缺失值
        features = features.dropna()
        
        return features, data
    
    def train_model(self, data):
        """训练Transformer模型"""
        # 准备特征
        features, data = self.prepare_features(data)
        
        # 计算目标变量（例如，5日后的收益率）
        data['target'] = data['close'].pct_change(5).shift(-5)
        data = data.dropna()
        
        # 确保特征和目标的长度一致
        min_length = min(len(features), len(data))
        features = features.iloc[:min_length]
        data = data.iloc[:min_length]
        
        # 分割训练集和测试集
        train_size = int(len(features) * 0.8)
        X_train = features.iloc[:train_size].values
        y_train = data['target'].iloc[:train_size].values
        X_val = features.iloc[train_size:].values
        y_val = data['target'].iloc[train_size:].values
        
        # 标准化特征
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 构建和训练模型
        self.transformer_model.build_model((X_train_scaled.shape[1],))
        history = self.transformer_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
        
        self.model_trained = True
        return history
    
    def generate_signals(self, data):
        """生成交易信号"""
        # 准备特征
        features, data = self.prepare_features(data)
        
        if not self.model_trained:
            # 如果模型未训练，使用传统策略
            return super().generate_signals(data)
        
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        
        # 使用Transformer模型预测
        predictions = self.transformer_model.predict(features_scaled)
        
        # 初始化信号
        data['Signal'] = 0
        
        # 基于预测结果生成信号
        # 只在有足够数据时生成信号
        if len(predictions) >= len(data):
            # 预测值大于0.005时买入
            buy_condition = predictions[:len(data)].flatten() > 0.005
            # 预测值小于-0.005时卖出
            sell_condition = predictions[:len(data)].flatten() < -0.005
            
            data.loc[buy_condition, 'Signal'] = 1
            data.loc[sell_condition, 'Signal'] = -1
        
        # 结合传统技术分析信号
        traditional_data = super().generate_signals(data)
        
        # 融合信号：只在传统信号和模型信号一致时交易
        data['Traditional_Signal'] = traditional_data['Signal']
        data['Signal'] = np.where(
            (data['Signal'] == data['Traditional_Signal']) & (data['Signal'] != 0),
            data['Signal'],
            0
        )
        
        return data
    
    def save_model(self, path):
        """保存模型"""
        if self.model_trained:
            self.transformer_model.save(path)
    
    def load_model(self, path):
        """加载模型"""
        self.transformer_model.load(path)
        self.model_trained = True
