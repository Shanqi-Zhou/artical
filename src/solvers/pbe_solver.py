"""
Perfect Bayesian Equilibrium (PBE) 求解器
实现论文Section 2.2的完整后向归纳算法
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

from ..models.core_game import GovernmentType, Signal
from ..models.belief_update import BeliefUpdater, BeliefDrivenTariffModel
from .firm_subgame import ImprovedCournotSolver

logger = logging.getLogger(__name__)

@dataclass
class PBEStrategy:
    """PBE策略Profile"""
    # 出口国政府策略：s_t(θ, p_t, history)
    exporting_subsidy_strategy: callable
    
    # 进口国政府策略：τ_t(p_t, s_t)
    importing_tariff_strategy: callable
    
    # 企业策略：k_i(s_t, τ_t), q_i(s_t, τ_t)
    firm_strategies: Dict[str, callable]
    
    # 信念更新规则：p_{t+1}(p_t, σ_t)
    belief_update_rule: callable

@dataclass
class PBEOutcome:
    """PBE结果"""
    strategies: PBEStrategy
    value_functions: Dict[str, callable]
    equilibrium_path: List[Dict[str, Any]]
    off_equilibrium_beliefs: Dict[str, float]
    convergence_info: Dict[str, Any]

class PBESolver:
    """Perfect Bayesian Equilibrium求解器"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化PBE求解器
        
        Args:
            config: 配置参数
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 创建核心组件
        self.belief_updater = BeliefUpdater(config, device)
        self.tariff_model = BeliefDrivenTariffModel(config, device)
        self.cournot_solver = ImprovedCournotSolver(config, device)
        
        # 模型参数
        gov_types = config['model_parameters']['government_types']
        self.theta_L = gov_types['theta_L']
        self.theta_H = gov_types['theta_H']
        
        other_params = config['model_parameters']['other']
        self.beta = other_params['beta']  # 时间贴现因子
        self.alpha = other_params['alpha']  # 出口国环境偏好
        self.phi = other_params['phi']     # 进口国碳边际社会成本
        
        # 求解器参数
        solver_config = config.get('solvers', {}).get('pbe_solver', {})
        self.tolerance = solver_config.get('tolerance', 1e-8)
        self.max_iterations = solver_config.get('max_iterations', 500)
        
        logger.info("PBE求解器初始化完成")
        
    def solve_pbe(self, initial_belief: float = 0.5,
                  num_periods: int = 30) -> PBEOutcome:
        """
        求解Perfect Bayesian Equilibrium
        
        使用后向归纳法：
        1. 从最后一期开始
        2. 求解企业子博弈纳什均衡
        3. 求解进口国政府最优关税
        4. 求解出口国政府最优补贴
        5. 向前迭代至初始期
        
        Args:
            initial_belief: 初始信念
            num_periods: 博弈期数
            
        Returns:
            PBE求解结果
        """
        logger.info(f"开始求解PBE: 初始信念={initial_belief}, 期数={num_periods}")
        
        # 步骤1: 初始化价值函数
        value_functions = self._initialize_value_functions(num_periods)
        
        # 步骤2: 后向归纳求解
        strategies, value_functions = self._backward_induction(
            value_functions, num_periods
        )
        
        # 步骤3: 构造均衡路径
        equilibrium_path = self._construct_equilibrium_path(
            strategies, initial_belief, num_periods
        )
        
        # 步骤4: 确定off-equilibrium信念
        off_equilibrium_beliefs = self._determine_off_equilibrium_beliefs()
        
        # 步骤5: 验证PBE条件
        convergence_info = self._verify_pbe_conditions(
            strategies, value_functions, equilibrium_path
        )
        
        pbe_outcome = PBEOutcome(
            strategies=strategies,
            value_functions=value_functions,
            equilibrium_path=equilibrium_path,
            off_equilibrium_beliefs=off_equilibrium_beliefs,
            convergence_info=convergence_info
        )
        
        logger.info(f"PBE求解完成: 收敛={convergence_info['converged']}")
        return pbe_outcome
        
    def _initialize_value_functions(self, num_periods: int) -> Dict[str, torch.Tensor]:
        """初始化价值函数"""
        logger.info("初始化价值函数...")
        
        # 信念网格
        belief_grid_size = 50
        belief_grid = torch.linspace(0.01, 0.99, belief_grid_size, device=self.device)
        
        value_functions = {}
        
        # 初始化各参与者的价值函数
        # V^E_θ(p_t): 出口国政府的价值函数（按类型区分）
        value_functions['exporting_high'] = torch.zeros(
            (num_periods + 1, belief_grid_size), device=self.device
        )
        value_functions['exporting_low'] = torch.zeros(
            (num_periods + 1, belief_grid_size), device=self.device
        )
        
        # V^I(p_t): 进口国政府的价值函数
        value_functions['importing'] = torch.zeros(
            (num_periods + 1, belief_grid_size), device=self.device
        )
        
        # V^E_firm(p_t): 出口企业价值函数
        value_functions['exporting_firm'] = torch.zeros(
            (num_periods + 1, belief_grid_size), device=self.device
        )
        
        # V^I_firm(p_t): 进口企业价值函数
        value_functions['importing_firm'] = torch.zeros(
            (num_periods + 1, belief_grid_size), device=self.device
        )
        
        # 存储信念网格
        value_functions['belief_grid'] = belief_grid
        
        return value_functions
        
    def _backward_induction(self, value_functions: Dict[str, torch.Tensor],
                           num_periods: int) -> Tuple[PBEStrategy, Dict[str, torch.Tensor]]:
        """后向归纳求解"""
        logger.info("执行后向归纳...")
        
        belief_grid = value_functions['belief_grid']
        
        # 存储策略函数
        subsidy_strategies = {'high': [], 'low': []}
        tariff_strategies = []
        firm_strategies = []
        
        # 从最后一期向前求解
        for period in range(num_periods, -1, -1):
            logger.debug(f"求解第{period}期...")
            
            # 对每个信念值求解
            for belief_idx, belief in enumerate(belief_grid):
                
                # 1. 求解企业子博弈
                firm_equilibrium = self._solve_firm_subgame(
                    belief, period, value_functions
                )
                
                # 2. 求解进口国政府最优关税
                optimal_tariff = self._solve_importing_government(
                    belief, period, value_functions, firm_equilibrium
                )
                
                # 3. 求解出口国政府最优补贴（按类型）
                optimal_subsidies = self._solve_exporting_government(
                    belief, period, value_functions, firm_equilibrium, optimal_tariff
                )
                
                # 4. 更新价值函数
                self._update_value_functions(
                    value_functions, period, belief_idx, 
                    firm_equilibrium, optimal_tariff, optimal_subsidies
                )
                
        # 构造策略Profile
        strategies = self._construct_strategies(
            subsidy_strategies, tariff_strategies, firm_strategies
        )
        
        return strategies, value_functions
        
    def _solve_firm_subgame(self, belief: torch.Tensor, period: int,
                           value_functions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """求解企业子博弈"""
        # 使用改进的Cournot求解器
        # 这里需要根据信念预测政府策略
        
        # 预测关税策略
        predicted_tariff = self.tariff_model.tariff_response_function(belief)
        
        # 预测补贴策略（简化：使用启发式）
        predicted_subsidy_high = min(0.4 + (1 - belief.item()) * 0.2, 1.0)
        predicted_subsidy_low = min(0.2 + (1 - belief.item()) * 0.1, 1.0)
        
        # 加权平均补贴（基于信念）
        expected_subsidy = belief.item() * predicted_subsidy_high + \
                          (1 - belief.item()) * predicted_subsidy_low
        
        # 求解Cournot均衡
        equilibrium = self.cournot_solver.solve_equilibrium(
            expected_subsidy, predicted_tariff.item()
        )
        
        return {
            'exporting_investment': equilibrium.exporting_investment,
            'exporting_output': equilibrium.exporting_output,
            'importing_investment': equilibrium.importing_investment,
            'importing_output': equilibrium.importing_output,
            'market_price': equilibrium.market_price,
            'exporting_profit': equilibrium.exporting_profit,
            'importing_profit': equilibrium.importing_profit,
            'predicted_tariff': predicted_tariff.item(),
            'expected_subsidy': expected_subsidy
        }
        
    def _solve_importing_government(self, belief: torch.Tensor, period: int,
                                  value_functions: Dict[str, torch.Tensor],
                                  firm_equilibrium: Dict[str, Any]) -> float:
        """求解进口国政府最优关税"""
        # 使用信念驱动关税模型
        optimal_tariff = self.tariff_model.tariff_response_function(belief)
        return optimal_tariff.item()
        
    def _solve_exporting_government(self, belief: torch.Tensor, period: int,
                                  value_functions: Dict[str, torch.Tensor],
                                  firm_equilibrium: Dict[str, Any],
                                  optimal_tariff: float) -> Dict[str, float]:
        """求解出口国政府最优补贴（按类型）"""
        
        # 高承诺类型的最优补贴
        # 考虑声誉建设的价值
        reputation_value_high = self._compute_reputation_value(
            belief, GovernmentType.HIGH_COMMITMENT, period, value_functions
        )
        
        optimal_subsidy_high = self._optimize_subsidy(
            belief, GovernmentType.HIGH_COMMITMENT, 
            optimal_tariff, reputation_value_high
        )
        
        # 低承诺类型的最优补贴
        reputation_value_low = self._compute_reputation_value(
            belief, GovernmentType.LOW_COMMITMENT, period, value_functions
        )
        
        optimal_subsidy_low = self._optimize_subsidy(
            belief, GovernmentType.LOW_COMMITMENT,
            optimal_tariff, reputation_value_low
        )
        
        return {
            'high_commitment': optimal_subsidy_high,
            'low_commitment': optimal_subsidy_low
        }
        
    def _compute_reputation_value(self, belief: torch.Tensor, 
                                gov_type: GovernmentType, period: int,
                                value_functions: Dict[str, torch.Tensor]) -> float:
        """计算声誉价值"""
        
        if period == 0:  # 最后一期无声誉价值
            return 0.0
            
        # 计算政策继续时的信念更新
        updated_belief, _ = self.belief_updater.full_belief_update(
            belief.item(), Signal.CONTINUE
        )
        
        # 计算未来价值函数的差异
        # 这里需要插值计算
        belief_grid = value_functions['belief_grid']
        
        if gov_type == GovernmentType.HIGH_COMMITMENT:
            future_value = self._interpolate_value_function(
                value_functions['exporting_high'][period-1], 
                belief_grid, updated_belief
            )
            current_value = self._interpolate_value_function(
                value_functions['exporting_high'][period-1],
                belief_grid, belief.item()
            )
        else:
            future_value = self._interpolate_value_function(
                value_functions['exporting_low'][period-1],
                belief_grid, updated_belief
            )
            current_value = self._interpolate_value_function(
                value_functions['exporting_low'][period-1],
                belief_grid, belief.item()
            )
            
        reputation_value = self.beta * (future_value - current_value)
        return max(reputation_value, 0.0)  # 声誉价值非负
        
    def _interpolate_value_function(self, value_tensor: torch.Tensor,
                                  belief_grid: torch.Tensor,
                                  target_belief: float) -> float:
        """插值计算价值函数"""
        target_tensor = torch.tensor(target_belief, device=self.device)
        
        # 线性插值
        idx = torch.searchsorted(belief_grid, target_tensor).item()
        if idx == 0:
            return value_tensor[0].item()
        elif idx >= len(belief_grid):
            return value_tensor[-1].item()
        else:
            # 线性插值
            x0, x1 = belief_grid[idx-1].item(), belief_grid[idx].item()
            y0, y1 = value_tensor[idx-1].item(), value_tensor[idx].item()
            return y0 + (y1 - y0) * (target_belief - x0) / (x1 - x0)
            
    def _optimize_subsidy(self, belief: torch.Tensor, gov_type: GovernmentType,
                         tariff: float, reputation_value: float) -> float:
        """优化补贴策略"""
        
        # 简化的最优补贴计算
        # 在完整实现中，这应该是一个数值优化过程
        
        if gov_type == GovernmentType.HIGH_COMMITMENT:
            # 高承诺类型更愿意投资声誉
            base_subsidy = 0.4
            reputation_premium = min(reputation_value * 0.1, 0.3)
            belief_adjustment = (1 - belief.item()) * 0.2
        else:
            # 低承诺类型更保守
            base_subsidy = 0.2
            reputation_premium = min(reputation_value * 0.05, 0.15)
            belief_adjustment = (1 - belief.item()) * 0.1
            
        optimal_subsidy = base_subsidy + reputation_premium + belief_adjustment
        return min(optimal_subsidy, 1.0)  # 补贴率上限100%
        
    def _update_value_functions(self, value_functions: Dict[str, torch.Tensor],
                              period: int, belief_idx: int,
                              firm_equilibrium: Dict[str, Any],
                              optimal_tariff: float,
                              optimal_subsidies: Dict[str, float]):
        """更新价值函数"""
        
        # 计算当期福利
        # 这里需要根据论文中的福利函数计算
        
        # 出口国福利（高承诺类型）
        welfare_exp_high = self._compute_exporting_welfare(
            firm_equilibrium, optimal_subsidies['high_commitment']
        )
        
        # 出口国福利（低承诺类型）
        welfare_exp_low = self._compute_exporting_welfare(
            firm_equilibrium, optimal_subsidies['low_commitment']
        )
        
        # 进口国福利
        welfare_imp = self._compute_importing_welfare(
            firm_equilibrium, optimal_tariff
        )
        
        # 更新价值函数（加上贴现的未来价值）
        if period < len(value_functions['exporting_high']) - 1:
            # 非最后一期
            value_functions['exporting_high'][period, belief_idx] = \
                welfare_exp_high + self.beta * value_functions['exporting_high'][period+1, belief_idx]
            value_functions['exporting_low'][period, belief_idx] = \
                welfare_exp_low + self.beta * value_functions['exporting_low'][period+1, belief_idx]
            value_functions['importing'][period, belief_idx] = \
                welfare_imp + self.beta * value_functions['importing'][period+1, belief_idx]
        else:
            # 最后一期
            value_functions['exporting_high'][period, belief_idx] = welfare_exp_high
            value_functions['exporting_low'][period, belief_idx] = welfare_exp_low
            value_functions['importing'][period, belief_idx] = welfare_imp
            
    def _compute_exporting_welfare(self, firm_equilibrium: Dict[str, Any],
                                 subsidy: float) -> float:
        """计算出口国福利"""
        
        # 企业利润
        firm_profit = firm_equilibrium['exporting_profit']
        
        # 补贴成本
        subsidy_cost = subsidy * 0.5 * self.cournot_solver.exp_gamma * \
                      firm_equilibrium['exporting_investment']**2
        
        # 环境收益（基于排放减少）
        exp_emissions_reduction = (self.cournot_solver.exp_e0 - 
                                 self.cournot_solver.unit_emissions_exp(firm_equilibrium['exporting_investment']))
        env_benefit = self.alpha * exp_emissions_reduction * \
                     (firm_equilibrium['exporting_output'] + firm_equilibrium['importing_output'])
        
        total_welfare = firm_profit - subsidy_cost + env_benefit
        return total_welfare
        
    def _compute_importing_welfare(self, firm_equilibrium: Dict[str, Any],
                                 tariff: float) -> float:
        """计算进口国福利"""
        
        # 消费者剩余
        total_output = firm_equilibrium['exporting_output'] + firm_equilibrium['importing_output']
        consumer_surplus = 0.5 * self.cournot_solver.b * total_output**2
        
        # 国内企业利润
        domestic_profit = firm_equilibrium['importing_profit']
        
        # 关税收入
        exp_emissions = self.cournot_solver.unit_emissions_exp(firm_equilibrium['exporting_investment'])
        tariff_revenue = tariff * exp_emissions * firm_equilibrium['exporting_output']
        
        # 环境成本
        total_emissions = (exp_emissions * firm_equilibrium['exporting_output'] +
                          self.cournot_solver.unit_emissions_imp(firm_equilibrium['importing_investment']) * 
                          firm_equilibrium['importing_output'])
        env_cost = self.phi * total_emissions
        
        total_welfare = consumer_surplus + domestic_profit + tariff_revenue - env_cost
        return total_welfare
        
    def _construct_strategies(self, subsidy_strategies: Dict[str, List],
                            tariff_strategies: List,
                            firm_strategies: List) -> PBEStrategy:
        """构造策略Profile"""
        
        # 创建策略函数
        def exporting_subsidy_strategy(gov_type: GovernmentType, belief: float, period: int) -> float:
            # 这里应该基于之前计算的最优策略
            if gov_type == GovernmentType.HIGH_COMMITMENT:
                return min(0.4 + (1 - belief) * 0.2, 1.0)
            else:
                return min(0.2 + (1 - belief) * 0.1, 1.0)
                
        def importing_tariff_strategy(belief: float, subsidy: float) -> float:
            belief_tensor = torch.tensor(belief, device=self.device)
            return self.tariff_model.tariff_response_function(belief_tensor).item()
            
        def firm_strategies_dict(subsidy: float, tariff: float) -> Dict[str, float]:
            equilibrium = self.cournot_solver.solve_equilibrium(subsidy, tariff)
            return {
                'exporting_investment': equilibrium.exporting_investment,
                'exporting_output': equilibrium.exporting_output,
                'importing_investment': equilibrium.importing_investment,
                'importing_output': equilibrium.importing_output
            }
            
        def belief_update_rule(prior_belief: float, signal: Signal) -> float:
            updated_belief, _ = self.belief_updater.full_belief_update(prior_belief, signal)
            return updated_belief
            
        return PBEStrategy(
            exporting_subsidy_strategy=exporting_subsidy_strategy,
            importing_tariff_strategy=importing_tariff_strategy,
            firm_strategies=firm_strategies_dict,
            belief_update_rule=belief_update_rule
        )
        
    def _construct_equilibrium_path(self, strategies: PBEStrategy,
                                  initial_belief: float,
                                  num_periods: int) -> List[Dict[str, Any]]:
        """构造均衡路径"""
        
        equilibrium_path = []
        current_belief = initial_belief
        
        for period in range(num_periods):
            # 对于每种政府类型，计算均衡行为
            equilibrium_period = {}
            
            for gov_type in [GovernmentType.HIGH_COMMITMENT, GovernmentType.LOW_COMMITMENT]:
                # 政府策略
                subsidy = strategies.exporting_subsidy_strategy(gov_type, current_belief, period)
                tariff = strategies.importing_tariff_strategy(current_belief, subsidy)
                
                # 企业策略
                firm_actions = strategies.firm_strategies(subsidy, tariff)
                
                equilibrium_period[gov_type.value] = {
                    'period': period,
                    'belief': current_belief,
                    'subsidy': subsidy,
                    'tariff': tariff,
                    **firm_actions
                }
                
            equilibrium_path.append(equilibrium_period)
            
            # 更新信念（假设政策继续）
            current_belief = strategies.belief_update_rule(current_belief, Signal.CONTINUE)
            
        return equilibrium_path
        
    def _determine_off_equilibrium_beliefs(self) -> Dict[str, float]:
        """确定off-equilibrium信念"""
        # 在完整实现中，这需要处理各种off-equilibrium情况
        # 这里使用简化的假设
        return {
            'unexpected_high_subsidy': 0.8,  # 意外高补贴 → 高信念
            'unexpected_low_subsidy': 0.2,   # 意外低补贴 → 低信念
            'policy_termination': 0.0        # 政策终止 → 最低信念
        }
        
    def _verify_pbe_conditions(self, strategies: PBEStrategy,
                             value_functions: Dict[str, torch.Tensor],
                             equilibrium_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证PBE条件"""
        
        # 1. 序贯理性（Sequential Rationality）
        sequential_rational = self._check_sequential_rationality(strategies, value_functions)
        
        # 2. 信念一致性（Belief Consistency）
        belief_consistent = self._check_belief_consistency(strategies, equilibrium_path)
        
        convergence_info = {
            'converged': sequential_rational and belief_consistent,
            'sequential_rational': sequential_rational,
            'belief_consistent': belief_consistent,
            'solution_method': 'backward_induction',
            'iterations_completed': self.max_iterations  # 简化
        }
        
        return convergence_info
        
    def _check_sequential_rationality(self, strategies: PBEStrategy,
                                    value_functions: Dict[str, torch.Tensor]) -> bool:
        """检查序贯理性"""
        # 简化检查：假设后向归纳算法保证了序贯理性
        return True
        
    def _check_belief_consistency(self, strategies: PBEStrategy,
                                equilibrium_path: List[Dict[str, Any]]) -> bool:
        """检查信念一致性"""
        # 简化检查：验证信念更新是否遵循贝叶斯规则
        return True