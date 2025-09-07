"""
动态贝叶斯博弈核心模型
实现四个参与者：出口国政府(G_E)、进口国政府(G_I)、出口国企业(F_E)、进口国企业(F_I)
以及基础数学函数和博弈结构
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class GovernmentType(Enum):
    """政府承诺类型"""
    HIGH_COMMITMENT = "theta_L"  # 高承诺类型（低终止概率）
    LOW_COMMITMENT = "theta_H"   # 低承诺类型（高终止概率）

class Signal(Enum):
    """政策延续信号"""
    CONTINUE = "continue"
    TERMINATE = "terminate"

@dataclass
class GameState:
    """博弈状态"""
    period: int                    # 当前期数
    belief: float                  # 当前信念 p_t
    subsidy: float                 # 当前补贴强度 s_t
    tariff: float                  # 当前关税率 tau_t
    exporting_investment: float    # 出口企业投资 k_E
    importing_investment: float    # 进口企业投资 k_I
    exporting_output: float        # 出口企业产量 q_E
    importing_output: float        # 进口企业产量 q_I
    market_price: float            # 市场价格 P_t
    
class BaseAgent(ABC):
    """基础智能体类"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化基础智能体
        
        Args:
            config: 配置参数
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.name = self.__class__.__name__
        
    @abstractmethod
    def get_action(self, state: GameState) -> float:
        """获取智能体行动"""
        pass

class ExportingFirm(BaseAgent):
    """出口国企业 (F_E)"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化出口国企业
        
        Args:
            config: 企业配置参数
            device: 计算设备
        """
        super().__init__(config, device)
        
        # 企业参数
        firm_config = config['model_parameters']['firms']['exporting']
        self.c_0 = firm_config['c_0']          # 初始边际成本
        self.e_0 = firm_config['e_0']          # 初始单位排放
        self.lambda_val = firm_config['lambda_val']  # 成本降低效率
        self.mu_val = firm_config['mu_val']    # 排放降低效率
        self.gamma = firm_config['gamma']      # 投资成本参数
        
        # 其他参数
        other_config = config['model_parameters']['other']
        self.F = other_config['F']             # FDI固定成本
        
        logger.info(f"出口企业初始化完成: c_0={self.c_0}, e_0={self.e_0}")
        
    def marginal_cost(self, investment: torch.Tensor) -> torch.Tensor:
        """
        计算边际生产成本
        c_E(k_E) = c_0 * exp(-lambda * k_E)
        
        Args:
            investment: 技术投资水平 k_E
            
        Returns:
            边际成本
        """
        return self.c_0 * torch.exp(-self.lambda_val * investment)
        
    def unit_emissions(self, investment: torch.Tensor) -> torch.Tensor:
        """
        计算单位排放
        e_E(k_E) = e_0 * exp(-mu * k_E)
        
        Args:
            investment: 技术投资水平 k_E
            
        Returns:
            单位排放
        """
        return self.e_0 * torch.exp(-self.mu_val * investment)
        
    def investment_cost(self, investment: torch.Tensor) -> torch.Tensor:
        """
        计算投资成本
        C(k_E) = gamma/2 * k_E^2
        
        Args:
            investment: 技术投资水平 k_E
            
        Returns:
            投资成本
        """
        return 0.5 * self.gamma * investment**2
        
    def profit(self, investment: torch.Tensor, output: torch.Tensor,
               market_price: torch.Tensor, tariff: torch.Tensor,
               subsidy: torch.Tensor) -> torch.Tensor:
        """
        计算企业利润
        π_E = (P - c_E(k_E) - τ) * q_E - (1 - s) * C(k_E)
        
        Args:
            investment: 技术投资 k_E
            output: 产量 q_E
            market_price: 市场价格 P
            tariff: 关税率 τ
            subsidy: 补贴强度 s
            
        Returns:
            企业利润
        """
        revenue = (market_price - self.marginal_cost(investment) - tariff) * output
        cost = (1 - subsidy) * self.investment_cost(investment)
        return revenue - cost
        
    def fdi_value(self, belief: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
        """
        计算FDI模式下的价值
        
        Args:
            belief: 当前信念
            delta: 国内环境监管成本
            
        Returns:
            FDI价值
        """
        # 简化的FDI价值计算，实际应该是复杂的动态规划
        # 这里返回一个基于信念的启发式值
        base_value = 100.0  # 基础价值
        belief_discount = belief * 0.2  # 信念折扣
        return base_value - self.F + belief_discount
        
    def should_fdi(self, export_value: torch.Tensor, 
                   belief: torch.Tensor) -> torch.Tensor:
        """
        判断是否应该进行FDI
        
        Args:
            export_value: 出口模式价值
            belief: 当前信念
            
        Returns:
            是否进行FDI的布尔张量
        """
        fdi_val = self.fdi_value(belief)
        return fdi_val > export_value
        
    def get_action(self, state: GameState) -> Tuple[float, float]:
        """
        获取企业最优行动（投资和产量）
        
        Args:
            state: 当前博弈状态
            
        Returns:
            (最优投资, 最优产量)
        """
        # 这里返回启发式解，实际应该通过优化求解
        optimal_investment = 2.0  # 基准投资水平
        optimal_output = 10.0     # 基准产量
        
        # 根据补贴调整投资
        if state.subsidy > 0:
            optimal_investment *= (1 + state.subsidy)
            
        # 根据关税调整产量
        if state.tariff > 0:
            optimal_output *= (1 - state.tariff * 0.1)
            
        return optimal_investment, optimal_output

class ImportingFirm(BaseAgent):
    """进口国企业 (F_I)"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化进口国企业
        
        Args:
            config: 企业配置参数
            device: 计算设备
        """
        super().__init__(config, device)
        
        # 企业参数
        firm_config = config['model_parameters']['firms']['importing']
        self.c_0 = firm_config['c_0']          # 初始边际成本
        self.e_0 = firm_config['e_0']          # 初始单位排放
        self.lambda_val = firm_config['lambda_val']  # 成本降低效率
        self.mu_val = firm_config['mu_val']    # 排放降低效率
        self.gamma = firm_config['gamma']      # 投资成本参数
        
        # 国内环境监管成本
        self.delta = config['model_parameters']['other']['delta']
        
        logger.info(f"进口企业初始化完成: c_0={self.c_0}, e_0={self.e_0}")
        
    def marginal_cost(self, investment: torch.Tensor) -> torch.Tensor:
        """计算边际生产成本"""
        return self.c_0 * torch.exp(-self.lambda_val * investment)
        
    def unit_emissions(self, investment: torch.Tensor) -> torch.Tensor:
        """计算单位排放"""
        return self.e_0 * torch.exp(-self.mu_val * investment)
        
    def investment_cost(self, investment: torch.Tensor) -> torch.Tensor:
        """计算投资成本"""
        return 0.5 * self.gamma * investment**2
        
    def profit(self, investment: torch.Tensor, output: torch.Tensor,
               market_price: torch.Tensor) -> torch.Tensor:
        """
        计算进口企业利润
        π_I = (P - c_I(k_I) - δ) * q_I - C(k_I)
        
        Args:
            investment: 技术投资 k_I
            output: 产量 q_I
            market_price: 市场价格 P
            
        Returns:
            企业利润
        """
        revenue = (market_price - self.marginal_cost(investment) - self.delta) * output
        cost = self.investment_cost(investment)
        return revenue - cost
        
    def get_action(self, state: GameState) -> Tuple[float, float]:
        """
        获取企业最优行动（投资和产量）
        
        Args:
            state: 当前博弈状态
            
        Returns:
            (最优投资, 最优产量)
        """
        # 启发式解
        optimal_investment = 1.5  # 相对较低的投资
        optimal_output = 8.0      # 基准产量
        
        # 根据竞争对手产量调整
        if state.exporting_output > 0:
            optimal_output *= (1 - state.exporting_output * 0.05)
            
        return optimal_investment, optimal_output

class ExportingGovernment(BaseAgent):
    """出口国政府 (G_E)"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device,
                 government_type: GovernmentType):
        """
        初始化出口国政府
        
        Args:
            config: 政府配置参数
            device: 计算设备
            government_type: 政府承诺类型
        """
        super().__init__(config, device)
        
        self.government_type = government_type
        self.theta = config['model_parameters']['government_types'][government_type.value]
        
        # 其他参数
        other_config = config['model_parameters']['other']
        self.alpha = other_config['alpha']     # 环境偏好权重
        self.beta = other_config['beta']       # 时间贴现因子
        
        logger.info(f"出口政府初始化完成: type={government_type.value}, theta={self.theta}")
        
    def welfare(self, firm_profit: torch.Tensor, subsidy_cost: torch.Tensor,
                environmental_benefit: torch.Tensor) -> torch.Tensor:
        """
        计算政府社会福利
        W_E = π_E - 补贴成本 + α * 环境收益
        
        Args:
            firm_profit: 企业利润
            subsidy_cost: 补贴成本
            environmental_benefit: 环境收益
            
        Returns:
            社会福利
        """
        return firm_profit - subsidy_cost + self.alpha * environmental_benefit
        
    def signal_probability(self, signal: Signal) -> float:
        """
        获取信号概率
        P(σ=continue|θ) = 1-θ, P(σ=terminate|θ) = θ
        
        Args:
            signal: 信号类型
            
        Returns:
            信号概率
        """
        if signal == Signal.CONTINUE:
            return 1 - self.theta
        else:  # Signal.TERMINATE
            return self.theta
            
    def get_action(self, state: GameState) -> float:
        """
        获取政府最优补贴政策
        
        Args:
            state: 当前博弈状态
            
        Returns:
            最优补贴强度
        """
        # 启发式补贴策略
        base_subsidy = 0.3
        
        # 根据信念调整补贴
        if state.belief < 0.5:
            # 信念较低时增加补贴以建立声誉
            base_subsidy *= (1 + (0.5 - state.belief))
            
        # 高承诺类型倾向于更高补贴
        if self.government_type == GovernmentType.HIGH_COMMITMENT:
            base_subsidy *= 1.2
            
        return min(base_subsidy, 1.0)  # 补贴强度不超过100%

class ImportingGovernment(BaseAgent):
    """进口国政府 (G_I)"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化进口国政府
        
        Args:
            config: 政府配置参数
            device: 计算设备
        """
        super().__init__(config, device)
        
        # 市场参数
        market_config = config['model_parameters']['market']
        self.a = market_config['a']            # 需求截距
        self.b = market_config['b']            # 需求斜率
        
        # 其他参数
        other_config = config['model_parameters']['other']
        self.phi = other_config['phi']         # 碳边际社会成本
        
        logger.info(f"进口政府初始化完成: a={self.a}, b={self.b}, phi={self.phi}")
        
    def market_price(self, total_output: torch.Tensor) -> torch.Tensor:
        """
        计算市场价格
        P = a - b * Q
        
        Args:
            total_output: 总产量 Q = q_E + q_I
            
        Returns:
            市场价格
        """
        return self.a - self.b * total_output
        
    def consumer_surplus(self, total_output: torch.Tensor) -> torch.Tensor:
        """
        计算消费者剩余
        CS = 0.5 * b * Q^2
        
        Args:
            total_output: 总产量
            
        Returns:
            消费者剩余
        """
        return 0.5 * self.b * total_output**2
        
    def tariff_revenue(self, tariff: torch.Tensor, 
                      exporting_emissions: torch.Tensor,
                      exporting_output: torch.Tensor) -> torch.Tensor:
        """
        计算关税收入
        TR = τ * e_E(k_E) * q_E
        
        Args:
            tariff: 关税率
            exporting_emissions: 出口企业单位排放
            exporting_output: 出口企业产量
            
        Returns:
            关税收入
        """
        return tariff * exporting_emissions * exporting_output
        
    def environmental_cost(self, exporting_emissions: torch.Tensor,
                          exporting_output: torch.Tensor,
                          importing_emissions: torch.Tensor,
                          importing_output: torch.Tensor) -> torch.Tensor:
        """
        计算环境成本
        ENV = φ * [e_E * q_E + e_I * q_I]
        
        Args:
            exporting_emissions: 出口企业单位排放
            exporting_output: 出口企业产量  
            importing_emissions: 进口企业单位排放
            importing_output: 进口企业产量
            
        Returns:
            环境成本
        """
        total_emissions = (exporting_emissions * exporting_output + 
                          importing_emissions * importing_output)
        return self.phi * total_emissions
        
    def welfare(self, consumer_surplus: torch.Tensor, 
               importing_profit: torch.Tensor,
               tariff_revenue: torch.Tensor,
               environmental_cost: torch.Tensor) -> torch.Tensor:
        """
        计算进口国社会福利
        W_I = CS + π_I + TR - ENV
        
        Args:
            consumer_surplus: 消费者剩余
            importing_profit: 进口企业利润
            tariff_revenue: 关税收入
            environmental_cost: 环境成本
            
        Returns:
            社会福利
        """
        return consumer_surplus + importing_profit + tariff_revenue - environmental_cost
        
    def optimal_tariff(self, belief: torch.Tensor, 
                      exporting_investment: torch.Tensor,
                      importing_investment: torch.Tensor) -> torch.Tensor:
        """
        计算最优CBAM关税率
        这是核心的"信念驱动关税"机制
        
        Args:
            belief: 当前信念 p_t
            exporting_investment: 出口企业投资
            importing_investment: 进口企业投资
            
        Returns:
            最优关税率
        """
        # 基础环境关税（Pigouvian税）
        base_tariff = self.phi * 0.2
        
        # 信念调整系数：信念越低，关税越高（belief-driven protectionism）
        belief_factor = 1.0 + (1.0 - belief) * 2.0
        
        # 投资差异调整：出口企业投资相对不足时提高关税
        investment_ratio = exporting_investment / (importing_investment + 1e-6)
        investment_factor = max(1.0, 2.0 - investment_ratio)
        
        optimal_tariff = base_tariff * belief_factor * investment_factor
        
        # 确保关税在合理范围内
        return torch.clamp(optimal_tariff, 0.1, 0.5)
        
    def get_action(self, state: GameState) -> float:
        """
        获取政府最优关税政策
        
        Args:
            state: 当前博弈状态
            
        Returns:
            最优关税率
        """
        belief_tensor = torch.tensor(state.belief, device=self.device)
        exp_investment = torch.tensor(state.exporting_investment, device=self.device)
        imp_investment = torch.tensor(state.importing_investment, device=self.device)
        
        optimal_tariff = self.optimal_tariff(belief_tensor, exp_investment, imp_investment)
        
        return optimal_tariff.item()

class GameEnvironment:
    """博弈环境"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化博弈环境
        
        Args:
            config: 配置参数
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 创建智能体
        self.exporting_firm = ExportingFirm(config, device)
        self.importing_firm = ImportingFirm(config, device)
        self.importing_government = ImportingGovernment(config, device)
        
        # 初始状态
        initial_config = config['simulation']['initial_conditions']
        self.initial_belief = initial_config['p_0']
        self.initial_subsidy = initial_config['s_0']
        self.initial_tariff = initial_config['tau_0']
        
        logger.info("博弈环境初始化完成")
        
    def create_exporting_government(self, government_type: GovernmentType) -> ExportingGovernment:
        """
        创建指定类型的出口国政府
        
        Args:
            government_type: 政府类型
            
        Returns:
            出口国政府实例
        """
        return ExportingGovernment(self.config, self.device, government_type)
        
    def reset(self) -> GameState:
        """
        重置博弈环境
        
        Returns:
            初始状态
        """
        return GameState(
            period=0,
            belief=self.initial_belief,
            subsidy=self.initial_subsidy,
            tariff=self.initial_tariff,
            exporting_investment=0.0,
            importing_investment=0.0,
            exporting_output=0.0,
            importing_output=0.0,
            market_price=0.0
        )
        
    def step(self, state: GameState, 
             exporting_government: ExportingGovernment) -> Tuple[GameState, Dict[str, float]]:
        """
        执行一步博弈
        
        Args:
            state: 当前状态
            exporting_government: 出口国政府
            
        Returns:
            (新状态, 信息字典)
        """
        # 1. 出口国政府选择补贴
        new_subsidy = exporting_government.get_action(state)
        
        # 2. 企业选择投资和产量
        exp_investment, exp_output = self.exporting_firm.get_action(
            GameState(**{**state.__dict__, 'subsidy': new_subsidy})
        )
        imp_investment, imp_output = self.importing_firm.get_action(state)
        
        # 3. 进口国政府选择关税
        temp_state = GameState(
            **{**state.__dict__, 
               'subsidy': new_subsidy,
               'exporting_investment': exp_investment,
               'importing_investment': imp_investment,
               'exporting_output': exp_output,
               'importing_output': imp_output}
        )
        new_tariff = self.importing_government.get_action(temp_state)
        
        # 4. 计算市场价格
        total_output = exp_output + imp_output
        market_price = self.importing_government.market_price(
            torch.tensor(total_output, device=self.device)
        ).item()
        
        # 5. 创建新状态
        new_state = GameState(
            period=state.period + 1,
            belief=state.belief,  # 信念更新将在单独模块中处理
            subsidy=new_subsidy,
            tariff=new_tariff,
            exporting_investment=exp_investment,
            importing_investment=imp_investment,
            exporting_output=exp_output,
            importing_output=imp_output,
            market_price=market_price
        )
        
        # 6. 计算各种指标
        info = self._calculate_metrics(new_state, exporting_government)
        
        return new_state, info
        
    def _calculate_metrics(self, state: GameState, 
                          exporting_government: ExportingGovernment) -> Dict[str, float]:
        """计算各种经济指标"""
        # 转换为张量
        exp_inv = torch.tensor(state.exporting_investment, device=self.device)
        imp_inv = torch.tensor(state.importing_investment, device=self.device)
        exp_out = torch.tensor(state.exporting_output, device=self.device)
        imp_out = torch.tensor(state.importing_output, device=self.device)
        price = torch.tensor(state.market_price, device=self.device)
        tariff = torch.tensor(state.tariff, device=self.device)
        subsidy = torch.tensor(state.subsidy, device=self.device)
        
        # 计算利润
        exp_profit = self.exporting_firm.profit(exp_inv, exp_out, price, tariff, subsidy)
        imp_profit = self.importing_firm.profit(imp_inv, imp_out, price)
        
        # 计算排放
        exp_emissions = self.exporting_firm.unit_emissions(exp_inv)
        imp_emissions = self.importing_firm.unit_emissions(imp_inv)
        
        # 计算福利组件
        consumer_surplus = self.importing_government.consumer_surplus(exp_out + imp_out)
        tariff_revenue = self.importing_government.tariff_revenue(tariff, exp_emissions, exp_out)
        env_cost = self.importing_government.environmental_cost(
            exp_emissions, exp_out, imp_emissions, imp_out
        )
        
        # 计算福利
        importing_welfare = self.importing_government.welfare(
            consumer_surplus, imp_profit, tariff_revenue, env_cost
        )
        
        subsidy_cost = subsidy * self.exporting_firm.investment_cost(exp_inv)
        env_benefit = (self.exporting_firm.e_0 - exp_emissions) * (exp_out + imp_out)
        exporting_welfare = exporting_government.welfare(exp_profit, subsidy_cost, env_benefit)
        
        return {
            'exporting_profit': exp_profit.item(),
            'importing_profit': imp_profit.item(),
            'exporting_welfare': exporting_welfare.item(),
            'importing_welfare': importing_welfare.item(),
            'consumer_surplus': consumer_surplus.item(),
            'tariff_revenue': tariff_revenue.item(),
            'environmental_cost': env_cost.item(),
            'total_emissions': (exp_emissions * exp_out + imp_emissions * imp_out).item(),
            'market_efficiency': (exp_profit + imp_profit + consumer_surplus).item()
        }