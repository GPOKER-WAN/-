import numpy as np
import pandas as pd
from .core import DepthStrategy
from models.model_manager import EnsembleModel

class EnsembleStrategy(DepthStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.ensemble_model = EnsembleModel(config)
        self.ensemble_enabled = config.get('model.ensemble.enabled', True)
        self.load_ensemble_models()
    
    def load_ensemble_models(self):
        """加载集成模型"""
        if self.ensemble_enabled:
            model_types = self.config.get('model.ensemble.models', ['rf', 'transformer'])
            self.ensemble_model.load_models(model_types, versions=['v1'])
            print("集成模型加载完成")
    
    def prepare_model_features(self, data):
        """准备模型特征"""
        # 计算所有技术指标
        data = self.calculate_indicators(data)
        
        # 选择特征列
        feature_columns = self.config.get('model.feature_columns', [
            'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200',
            'MA5_diff', 'MA10_diff', 'MA20_diff',
            'volatility_5d', 'volatility_20d',
            'momentum_5', 'momentum_10', 'momentum_20',
            'volume_ratio', 'price_change_pct'
        ])
        
        # 确保所有特征列存在
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 计算额外的特征
        if 'volatility_5d' not in data.columns:
            data['volatility_5d'] = data['close'].rolling(window=5).std()
        if 'volatility_20d' not in data.columns:
            data['volatility_20d'] = data['close'].rolling(window=20).std()
        if 'momentum_5' not in data.columns:
            data['momentum_5'] = data['close'].pct_change(5)
        if 'momentum_10' not in data.columns:
            data['momentum_10'] = data['close'].pct_change(10)
        if 'momentum_20' not in data.columns:
            data['momentum_20'] = data['close'].pct_change(20)
        if 'volume_ratio' not in data.columns:
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
        if 'price_change_pct' not in data.columns:
            data['price_change_pct'] = data['close'].pct_change()
        if 'MA5' not in data.columns:
            data['MA5'] = data['close'].rolling(window=5).mean()
        if 'MA10' not in data.columns:
            data['MA10'] = data['close'].rolling(window=10).mean()
        if 'MA5_diff' not in data.columns:
            data['MA5_diff'] = data['MA5'] - data['MA5'].shift(1)
        if 'MA10_diff' not in data.columns:
            data['MA10_diff'] = data['MA10'] - data['MA10'].shift(1)
        if 'MA20_diff' not in data.columns:
            data['MA20_diff'] = data['MA20'] - data['MA20'].shift(1)
        
        # 更新可用特征
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 提取特征
        X = data[available_features].dropna()
        
        return X, available_features
    
    def generate_signals_with_models(self, data):
        """使用多模型生成交易信号"""
        # 准备数据
        X, features = self.prepare_model_features(data)
        
        if len(X) == 0:
            data['Model_Signal'] = 0
            return data
        
        # 使用集成模型预测
        if self.ensemble_enabled:
            result = self.ensemble_model.ensemble_predict(X)
            predictions = result['final_prediction']
            probabilities = result['final_probability']
        else:
            # 回退到传统策略
            data = self.generate_signals(data)
            data['Model_Signal'] = data['Signal']
            return data
        
        # 将预测结果映射回原始数据
        data['Model_Signal'] = 0
        data.loc[X.index, 'Model_Signal'] = predictions
        data['Model_Probability'] = 0
        data.loc[X.index, 'Model_Probability'] = probabilities
        
        return data
    
    def generate_signals(self, data):
        """生成交易信号（结合多模型和传统策略）"""
        # 先使用传统策略生成信号
        data = super().generate_signals(data)
        
        # 再使用多模型策略生成信号
        data = self.generate_signals_with_models(data)
        
        # 结合两种信号
        data['Combined_Signal'] = 0
        
        # 买入信号：两种策略都看涨
        buy_condition = (data['Signal'] == 1) & (data['Model_Signal'] == 1) & (data.get('Model_Probability', 0) > 0.6)
        
        # 卖出信号：两种策略都看跌
        sell_condition = (data['Signal'] == -1) & (data['Model_Signal'] == 0) & (data.get('Model_Probability', 1) < 0.4)
        
        data.loc[buy_condition, 'Combined_Signal'] = 1
        data.loc[sell_condition, 'Combined_Signal'] = -1
        
        return data
    
    def apply_filters(self, data):
        """应用过滤器"""
        # 先应用传统过滤器
        data = super().apply_filters(data)
        
        # 添加模型概率过滤器
        if 'Model_Probability' in data.columns:
            # 只有当模型概率足够高时才交易
            probability_filter = (data['Model_Probability'] > 0.6) | (data['Model_Probability'] < 0.4)
            data.loc[~probability_filter, 'Filtered_Signal'] = 0
        
        return data
    
    def update_models(self):
        """更新模型"""
        # 重新加载模型
        self.load_ensemble_models()
        print("模型已更新")
    
    def get_model_performance(self):
        """获取模型性能"""
        # 这里可以添加模型性能评估逻辑
        # 例如，使用历史数据评估模型性能
        return {}
