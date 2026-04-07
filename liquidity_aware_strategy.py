import numpy as np
import pandas as pd
from .core import DepthStrategy

class LiquidityAwareStrategy(DepthStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.max_position_pct = config.get('max_position_pct', 0.1)  # 最大持仓占当日成交量比例
        self.max_slippage_pct = config.get('max_slippage_pct', 0.02)  # 最大可接受滑点百分比
        self.enable_batch_sell = config.get('enable_batch_sell', True)  # 是否启用分批卖出
        self.use_limit_order = config.get('use_limit_order', True)  # 是否使用限价单
        
    def check_liquidity(self, position_size, daily_volume):
        """检查流动性是否充足"""
        position_pct = position_size / daily_volume if daily_volume > 0 else 1.0
        is_liquid = position_pct <= self.max_position_pct
        return is_liquid, position_pct
    
    def calculate_batch_sizes(self, position_size, daily_volume):
        """计算分批卖出的批次"""
        if not self.enable_batch_sell:
            return [position_size]
        
        max_per_batch = daily_volume * self.max_position_pct
        if position_size <= max_per_batch:
            return [position_size]
        
        batches = []
        remaining = position_size
        while remaining > 0:
            batch_size = min(remaining, max_per_batch)
            batches.append(batch_size)
            remaining -= batch_size
        
        return batches
    
    def calculate_limit_price(self, target_price, is_sell, atr):
        """计算限价单价格"""
        if not self.use_limit_order:
            return target_price
        
        # 对于卖出，限价单价格略低于目标价，确保成交
        # 对于买入，限价单价格略高于目标价
        slippage_buffer = atr * 0.1  # 使用ATR的10%作为缓冲
        
        if is_sell:
            limit_price = target_price - slippage_buffer
        else:
            limit_price = target_price + slippage_buffer
        
        return limit_price
    
    def generate_signals(self, data):
        """生成交易信号，增加流动性检查"""
        data = super().generate_signals(data)
        
        # 添加流动性指标
        data['liquidity_score'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # 在生成信号时考虑流动性
        # 只在流动性充足时生成买入信号
        liquidity_filter = data['liquidity_score'] > 0.8  # 成交量不低于20日均值的80%
        
        # 调整信号
        data.loc[~liquidity_filter & (data['Signal'] == 1), 'Signal'] = 0
        
        return data
    
    def get_stop_loss_take_profit(self, entry_price, atr, is_long):
        """计算止损和止盈价格，考虑流动性"""
        stop_loss, take_profit = super().get_stop_loss_take_profit(entry_price, atr, is_long)
        
        # 根据流动性调整止损止盈距离
        # 如果流动性较差，适当放宽止损，避免过早触发
        # 这里可以根据实际情况调整
        
        return stop_loss, take_profit
    
    def execute_sell_with_liquidity_check(self, position, current_price, stop_loss, take_profit, daily_volume, atr):
        """执行卖出操作，考虑流动性"""
        # 检查是否触发止损或止盈
        is_stop_loss = current_price <= stop_loss
        is_take_profit = is_stop_loss == False and current_price >= take_profit
        
        if not (is_stop_loss or is_take_profit):
            return None
        
        target_price = stop_loss if is_stop_loss else take_profit
        
        # 检查流动性
        is_liquid, position_pct = self.check_liquidity(position, daily_volume)
        
        # 计算滑点
        actual_slippage = abs(current_price - target_price)
        slippage_pct = actual_slippage / target_price
        
        # 如果滑点过大，使用限价单
        if slippage_pct > self.max_slippage_pct:
            limit_price = self.calculate_limit_price(target_price, True, atr)
            execution_price = limit_price
            execution_type = 'limit_order'
        else:
            execution_price = current_price
            execution_type = 'market_order'
        
        # 如果流动性不足，分批卖出
        if not is_liquid:
            batches = self.calculate_batch_sizes(position, daily_volume)
            return {
                'type': 'batch_sell',
                'batches': batches,
                'execution_price': execution_price,
                'target_price': target_price,
                'is_stop_loss': is_stop_loss,
                'position_pct': position_pct,
                'slippage': actual_slippage,
                'slippage_pct': slippage_pct,
                'execution_type': execution_type
            }
        else:
            return {
                'type': 'single_sell',
                'position': position,
                'execution_price': execution_price,
                'target_price': target_price,
                'is_stop_loss': is_stop_loss,
                'position_pct': position_pct,
                'slippage': actual_slippage,
                'slippage_pct': slippage_pct,
                'execution_type': execution_type
            }
