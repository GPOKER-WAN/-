import numpy as np
import pandas as pd
from scipy.stats import zscore

class DepthStrategy:
    def __init__(self, config):
        self.config = config
        self.indicators = {}
        self.signals = {}
    
    def calculate_indicators(self, data):
        """计算所有技术指标"""
        # 计算ATR (Average True Range)
        data['ATR'] = self._calculate_atr(data, 14)
        
        # 计算移动平均线
        data['MA20'] = data['close'].rolling(window=20).mean()
        data['MA60'] = data['close'].rolling(window=60).mean()
        data['MA120'] = data['close'].rolling(window=120).mean()
        
        # 计算布林带
        data['BB_Middle'] = data['close'].rolling(window=20).mean()
        data['BB_Std'] = data['close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']
        
        # 计算RSI
        data['RSI'] = self._calculate_rsi(data, 14)
        
        # 计算MACD
        data['MACD'], data['Signal'], data['Histogram'] = self._calculate_macd(data)
        
        # 计算趋势强度
        data['Trend_Strength'] = self._calculate_trend_strength(data)
        
        # 增加成交量因子
        data['Volume_MA20'] = data['volume'].rolling(window=20).mean()
        data['Volume_MA60'] = data['volume'].rolling(window=60).mean()
        data['Volume_Ratio'] = data['volume'] / data['Volume_MA20']
        data['Volume_Change'] = data['volume'].pct_change()
        
        # 增加情绪因子（基于价格动量）
        data['Price_Momentum'] = data['close'].pct_change(10)
        data['Price_Volatility'] = data['close'].rolling(window=10).std() / data['close'].rolling(window=10).mean()
        
        # 增加其他技术指标
        data['K'], data['D'] = self._calculate_stochastics(data)
        data['ADX'] = self._calculate_adx(data)
        
        return data
    
    def _calculate_atr(self, data, period):
        """计算ATR"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        true_range = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
        return true_range.rolling(window=period).mean()
    
    def _calculate_rsi(self, data, period):
        """计算RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD"""
        ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def _calculate_trend_strength(self, data):
        """计算趋势强度"""
        # 基于移动平均线的趋势强度
        short_term = data['MA20'] - data['MA60']
        long_term = data['MA60'] - data['MA120']
        trend_strength = (short_term + long_term) / data['ATR']
        return trend_strength
    
    def _calculate_stochastics(self, data, period=14, smooth_k=3, smooth_d=3):
        """计算随机指标"""
        high = data['high'].rolling(window=period).max()
        low = data['low'].rolling(window=period).min()
        k = ((data['close'] - low) / (high - low)) * 100
        k = k.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        return k, d
    
    def _calculate_adx(self, data, period=14):
        """计算平均方向指数"""
        # 计算上升和下降趋势
        up_move = data['high'] - data['high'].shift(1)
        down_move = data['low'].shift(1) - data['low']
        
        # 过滤掉负值
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # 计算真实范围
        true_range = np.maximum(data['high'] - data['low'], 
                               np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                          abs(data['low'] - data['close'].shift(1))))
        
        # 计算平均真实范围
        atr = true_range.rolling(window=period).mean()
        
        # 计算方向指标
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # 计算方向指数
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def generate_signals(self, data):
        """生成交易信号"""
        data = self.calculate_indicators(data)
        
        # 初始化信号
        data['Signal'] = 0
        
        # 多时间框架确认（模拟，实际应用中需要不同时间框架的数据）
        # 这里使用同一数据模拟不同时间框架的确认
        
        # 趋势大师视角：多时间框架趋势确认
        trend_condition = (data['MA20'] > data['MA60']) & (data['MA60'] > data['MA120']) & (data['Trend_Strength'] > 1.0)
        
        # 波动率专家视角：布林带突破
        volatility_condition = (data['close'] > data['BB_Upper']) & (data['BB_Std'] < data['BB_Std'].rolling(20).mean())
        
        # 风险控制视角：RSI在合理范围
        rsi_condition = (data['RSI'] < 65) & (data['RSI'] > 35)
        
        # MACD金叉
        macd_condition = (data['MACD'] > data['Signal']) & (data['MACD'] > 0)
        
        # 成交量因子：成交量放大
        volume_condition = (data['Volume_Ratio'] > 1.2) & (data['Volume_MA20'] > data['Volume_MA60'])
        
        # 情绪因子：价格动量为正
        momentum_condition = data['Price_Momentum'] > 0
        
        # 其他技术指标：随机指标金叉，ADX显示趋势
        stochastics_condition = (data['K'] > data['D']) & (data['K'] < 80)
        adx_condition = data['ADX'] > 25
        
        # 综合信号（放宽条件）
        buy_condition = trend_condition & volatility_condition & rsi_condition & macd_condition & (data['Volume_Ratio'] > 1.1) & momentum_condition & (data['ADX'] > 20)
        sell_condition = (~trend_condition) & (data['close'] < data['BB_Lower']) & (data['RSI'] > 30) & (data['MACD'] < data['Signal']) & (data['Price_Momentum'] < 0) & (data['ADX'] > 20)
        
        data.loc[buy_condition, 'Signal'] = 1
        data.loc[sell_condition, 'Signal'] = -1
        
        return data
    
    def apply_filters(self, data):
        """应用过滤器"""
        # 时间过滤器：避免在市场开盘和收盘的高波动期交易
        # 这里需要根据实际交易时间进行调整
        
        # 波动率过滤器：避免在极端高波动时期交易
        volatility_filter = data['ATR'] < data['ATR'].rolling(20).mean() * 2
        
        # 趋势强度过滤器：只在趋势明确时交易
        trend_strength_filter = abs(data['Trend_Strength']) > 0.5
        
        # 应用过滤器
        data['Filtered_Signal'] = data['Signal']
        data.loc[~volatility_filter, 'Filtered_Signal'] = 0
        data.loc[~trend_strength_filter, 'Filtered_Signal'] = 0
        
        return data
    
    def calculate_position_size(self, capital, atr, win_rate, risk_reward):
        """使用凯利公式计算仓位大小"""
        # 凯利公式: K = (W * R - L) / R
        # W: 胜率, R: 风险回报比, L: 败率 = 1 - W
        kelly_percentage = (win_rate * risk_reward - (1 - win_rate)) / risk_reward
        kelly_percentage = max(0, min(0.2, kelly_percentage))  # 限制在0-20%之间，更保守
        
        # 基于ATR的风险控制
        risk_per_trade = capital * 0.015  # 每笔交易风险控制在1.5%，更保守
        position_size = risk_per_trade / atr
        
        # 结合凯利公式调整
        kelly_position = capital * kelly_percentage
        position_size = min(position_size, kelly_position)
        
        return position_size
    
    def get_stop_loss_take_profit(self, entry_price, atr, is_long):
        """计算止损和止盈价格"""
        # 更严格的止损设置，控制最大回撤
        if is_long:
            stop_loss = entry_price - 1.5 * atr  # 更严格的止损
            take_profit = entry_price + 2.5 * atr  # 调整止盈
        else:
            stop_loss = entry_price + 1.5 * atr  # 更严格的止损
            take_profit = entry_price - 2.5 * atr  # 调整止盈
        
        return stop_loss, take_profit