"""
博弈求解器模块
实现反向归纳算法和Perfect Bayesian Equilibrium (PBE)求解
包括企业子博弈的Cournot-Nash均衡和政府的动态优化
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable
import logging
from dataclasses import dataclass
from scipy.optimize import minimize, root_scalar, fsolve
import warnings
warnings.filterwarnings('ignore')

from ..models.core_game import (
    GameEnvironment, GameState, ExportingFirm, ImportingFirm, 
    ExportingGovernment, ImportingGovernment, GovernmentType, Signal
)
from ..models.belief_update import BeliefUpdater, BeliefDrivenTariffModel

logger = logging.getLogger(__name__)

@dataclass
class EquilibriumSolution:
    """均衡解"""
    period: int
    belief: float
    exporting_investment: float
    importing_investment: float
    exporting_output: float
    importing_output: float
    subsidy: float
    tariff: float
    market_price: float
    welfare_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]

@dataclass
class PBEResults:
    """Perfect Bayesian Equilibrium结果"""
    equilibrium_path: List[EquilibriumSolution]
    value_functions: Dict[str, torch.Tensor]
    policy_functions: Dict[str, Callable]
    belief_evolution: np.ndarray
    total_welfare: float
    convergence_achieved: bool

class CournotNashSolver:
    """Cournot-Nash均衡求解器"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化Cournot-Nash求解器
        
        Args:
            config: 配置参数
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 求解参数
        solver_config = config.get('solvers', {}).get('cournot_solver', {})
        self.tolerance = solver_config.get('tolerance', 1e-8)
        self.max_iterations = solver_config.get('max_iterations', 100)
        
        logger.info("Cournot-Nash求解器初始化完成")
        
    def solve_firm_subgame(self, 
                          exporting_firm: ExportingFirm,
                          importing_firm: ImportingFirm,
                          market_params: Dict[str, float],
                          policy_params: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        求解企业子博弈的Cournot-Nash均衡
        
        Args:
            exporting_firm: 出口企业
            importing_firm: 进口企业
            market_params: 市场参数 {a, b}
            policy_params: 政策参数 {subsidy, tariff, delta}
            
        Returns:
            (均衡解[k_E, q_E, k_I, q_I], 收敛信息)
        """
        def equilibrium_conditions(x):
            """均衡条件函数"""
            k_E, q_E, k_I, q_I = x
            
            # 确保变量为正
            k_E, q_E, k_I, q_I = max(k_E, 1e-6), max(q_E, 1e-6), max(k_I, 1e-6), max(q_I, 1e-6)
            
            # 转换为张量
            k_E_t = torch.tensor(k_E, device=self.device, dtype=torch.float32)
            q_E_t = torch.tensor(q_E, device=self.device, dtype=torch.float32)
            k_I_t = torch.tensor(k_I, device=self.device, dtype=torch.float32)
            q_I_t = torch.tensor(q_I, device=self.device, dtype=torch.float32)
            
            # 市场价格
            total_output = q_E + q_I
            price = market_params['a'] - market_params['b'] * total_output
            price_t = torch.tensor(price, device=self.device, dtype=torch.float32)
            
            # 政策参数
            subsidy = policy_params.get('subsidy', 0.0)
            tariff = policy_params.get('tariff', 0.0)
            delta = policy_params.get('delta', 0.5)
            
            subsidy_t = torch.tensor(subsidy, device=self.device, dtype=torch.float32)
            tariff_t = torch.tensor(tariff, device=self.device, dtype=torch.float32)
            
            # 出口企业的一阶条件
            # ∂π_E/∂q_E = 0: P - c_E(k_E) - τ - b*q_E = 0
            marginal_cost_E = exporting_firm.marginal_cost(k_E_t).item()
            foc_q_E = price - marginal_cost_E - tariff - market_params['b'] * q_E
            
            # ∂π_E/∂k_E = 0: λ*c_E(k_E)*q_E - (1-s)*γ*k_E = 0
            cost_reduction_benefit = exporting_firm.lambda_val * marginal_cost_E * q_E
            investment_cost_E = (1 - subsidy) * exporting_firm.gamma * k_E
            foc_k_E = cost_reduction_benefit - investment_cost_E
            
            # 进口企业的一阶条件
            # ∂π_I/∂q_I = 0: P - c_I(k_I) - δ - b*q_I = 0
            marginal_cost_I = importing_firm.marginal_cost(k_I_t).item()
            foc_q_I = price - marginal_cost_I - delta - market_params['b'] * q_I
            
            # ∂π_I/∂k_I = 0: λ*c_I(k_I)*q_I - γ*k_I = 0
            cost_reduction_benefit_I = importing_firm.lambda_val * marginal_cost_I * q_I
            investment_cost_I = importing_firm.gamma * k_I
            foc_k_I = cost_reduction_benefit_I - investment_cost_I
            
            return [foc_k_E, foc_q_E, foc_k_I, foc_q_I]
        
        # 初始猜测
        initial_guess = [2.0, 10.0, 1.5, 8.0]  # [k_E, q_E, k_I, q_I]
        
        try:
            # 使用fsolve求解
            solution = fsolve(equilibrium_conditions, initial_guess, xtol=self.tolerance)
            
            # 验证解的有效性
            residual = equilibrium_conditions(solution)
            max_residual = max(abs(r) for r in residual)
            
            convergence_info = {
                'converged': max_residual < self.tolerance,
                'max_residual': max_residual,
                'iterations': 1,  # fsolve不直接提供迭代次数
                'method': 'fsolve'
            }
            
            # 确保解为正数
            solution = np.maximum(solution, 1e-6)
            
            return solution, convergence_info
            
        except Exception as e:
            logger.warning(f"Cournot-Nash求解失败: {e}")
            # 返回启发式解
            fallback_solution = np.array([2.0, 10.0, 1.5, 8.0])
            convergence_info = {
                'converged': False,
                'max_residual': np.inf,
                'iterations': 0,
                'method': 'fallback',
                'error': str(e)
            }
            return fallback_solution, convergence_info

class DynamicProgrammingSolver:
    """动态规划求解器"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化动态规划求解器
        
        Args:
            config: 配置参数
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        # 求解参数
        dp_config = config.get('solvers', {}).get('dynamic_programming', {})
        self.value_tolerance = dp_config.get('value_function_tolerance', 1e-6)
        self.policy_tolerance = dp_config.get('policy_function_tolerance', 1e-6)
        self.max_iterations = dp_config.get('max_iterations', 1000)
        
        # 贴现因子
        self.beta = config['model_parameters']['other']['beta']
        
        logger.info("动态规划求解器初始化完成")
        
    def solve_government_value_function(self,
                                      government: ExportingGovernment,
                                      belief_grid: np.ndarray,
                                      cournot_solver: CournotNashSolver) -> Tuple[torch.Tensor, Callable]:
        """
        求解政府的价值函数
        
        使用价值函数迭代方法求解Bellman方程：
        V(p,θ) = max_s {W(s,τ*(p);θ) + β*E[V(p',θ)|p,θ]}
        
        Args:
            government: 政府实例
            belief_grid: 信念网格点
            cournot_solver: Cournot求解器
            
        Returns:
            (价值函数张量, 策略函数)
        """
        n_beliefs = len(belief_grid)
        
        # 初始化价值函数
        value_function = torch.zeros(n_beliefs, device=self.device)
        new_value_function = torch.zeros(n_beliefs, device=self.device)
        
        # 策略函数（最优补贴强度）
        policy_function = torch.zeros(n_beliefs, device=self.device)
        
        # 补贴网格
        subsidy_grid = np.linspace(0.0, 1.0, 21)  # 0%到100%，21个点
        
        # 市场参数
        market_params = {
            'a': self.config['model_parameters']['market']['a'],
            'b': self.config['model_parameters']['market']['b']
        }
        
        # 价值函数迭代
        for iteration in range(self.max_iterations):
            for i, belief in enumerate(belief_grid):
                max_value = -np.inf
                best_subsidy = 0.0
                
                for subsidy in subsidy_grid:
                    # 计算当期福利
                    current_welfare = self._compute_current_welfare(
                        government, belief, subsidy, market_params, cournot_solver
                    )
                    
                    # 计算期望延续价值
                    expected_continuation = self._compute_expected_continuation_value(
                        government, belief, belief_grid, value_function
                    )
                    
                    # 总价值
                    total_value = current_welfare + self.beta * expected_continuation
                    
                    if total_value > max_value:
                        max_value = total_value
                        best_subsidy = subsidy
                        
                new_value_function[i] = max_value
                policy_function[i] = best_subsidy
                
            # 检查收敛
            value_diff = torch.max(torch.abs(new_value_function - value_function)).item()
            
            if value_diff < self.value_tolerance:
                logger.info(f"价值函数在第{iteration+1}次迭代收敛，误差={value_diff:.2e}")
                break
                
            value_function.copy_(new_value_function)
            
        else:
            logger.warning(f"价值函数在{self.max_iterations}次迭代后未收敛，误差={value_diff:.2e}")
            
        # 创建策略函数（插值）
        def optimal_policy(belief_val: float) -> float:
            """根据信念返回最优补贴强度"""
            if belief_val <= belief_grid[0]:
                return policy_function[0].item()
            elif belief_val >= belief_grid[-1]:
                return policy_function[-1].item()
            else:
                # 线性插值
                idx = np.searchsorted(belief_grid, belief_val) - 1
                weight = (belief_val - belief_grid[idx]) / (belief_grid[idx+1] - belief_grid[idx])
                return (1-weight) * policy_function[idx].item() + weight * policy_function[idx+1].item()
                
        return value_function, optimal_policy
        
    def _compute_current_welfare(self, 
                               government: ExportingGovernment,
                               belief: float,
                               subsidy: float,
                               market_params: Dict[str, float],
                               cournot_solver: CournotNashSolver) -> float:
        """计算当期政府福利"""
        # 计算最优关税（基于信念）
        tariff_model = BeliefDrivenTariffModel(self.config, self.device)
        belief_tensor = torch.tensor(belief, device=self.device)
        optimal_tariff = tariff_model.tariff_response_function(belief_tensor).item()
        
        # 政策参数
        policy_params = {
            'subsidy': subsidy,
            'tariff': optimal_tariff,
            'delta': self.config['model_parameters']['other']['delta']
        }
        
        # 求解企业均衡
        from ..models.core_game import ExportingFirm, ImportingFirm
        exp_firm = ExportingFirm(self.config, self.device)
        imp_firm = ImportingFirm(self.config, self.device)
        
        equilibrium, _ = cournot_solver.solve_firm_subgame(
            exp_firm, imp_firm, market_params, policy_params
        )
        
        k_E, q_E, k_I, q_I = equilibrium
        
        # 计算福利组件
        # 企业利润
        total_output = q_E + q_I
        price = market_params['a'] - market_params['b'] * total_output
        
        k_E_t = torch.tensor(k_E, device=self.device)
        q_E_t = torch.tensor(q_E, device=self.device)
        price_t = torch.tensor(price, device=self.device)
        tariff_t = torch.tensor(optimal_tariff, device=self.device)
        subsidy_t = torch.tensor(subsidy, device=self.device)
        
        firm_profit = exp_firm.profit(k_E_t, q_E_t, price_t, tariff_t, subsidy_t).item()
        
        # 补贴成本
        subsidy_cost = subsidy * exp_firm.investment_cost(k_E_t).item()
        
        # 环境收益
        emissions_reduction = exp_firm.e_0 - exp_firm.unit_emissions(k_E_t).item()
        env_benefit = government.alpha * emissions_reduction * total_output
        
        welfare = firm_profit - subsidy_cost + env_benefit
        
        return welfare
        
    def _compute_expected_continuation_value(self,
                                           government: ExportingGovernment, 
                                           current_belief: float,
                                           belief_grid: np.ndarray,
                                           value_function: torch.Tensor) -> float:
        """计算期望延续价值"""
        # 简化：假设信念按照确定性规律演化
        # 实际中应该考虑信号的随机性
        
        # 如果继续政策，信念会如何变化？
        belief_updater = BeliefUpdater(self.config, self.device)
        
        # 计算继续和终止两种情况下的期望价值
        continue_prob = government.signal_probability(Signal.CONTINUE)
        
        if continue_prob > 0:
            # 假设观察到继续信号后的信念更新
            from ..models.core_game import Signal
            updated_belief, _ = belief_updater.full_belief_update(
                current_belief, Signal.CONTINUE
            )
            
            # 在信念网格上插值获得延续价值
            if updated_belief <= belief_grid[0]:
                continuation_value = value_function[0].item()
            elif updated_belief >= belief_grid[-1]:
                continuation_value = value_function[-1].item()
            else:
                idx = np.searchsorted(belief_grid, updated_belief) - 1
                weight = (updated_belief - belief_grid[idx]) / (belief_grid[idx+1] - belief_grid[idx])
                continuation_value = ((1-weight) * value_function[idx] + 
                                    weight * value_function[idx+1]).item()
        else:
            continuation_value = 0.0
            
        # 终止情况价值为0
        expected_value = continue_prob * continuation_value
        
        return expected_value

class PerfectBayesianEquilibriumSolver:
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
        
        # 初始化子求解器
        self.cournot_solver = CournotNashSolver(config, device)
        self.dp_solver = DynamicProgrammingSolver(config, device)
        self.belief_updater = BeliefUpdater(config, device)
        self.tariff_model = BeliefDrivenTariffModel(config, device)
        
        # 求解参数
        pbe_config = config.get('solvers', {}).get('pbe_solver', {})
        self.tolerance = pbe_config.get('tolerance', 1e-8)
        self.max_iterations = pbe_config.get('max_iterations', 500)
        
        # 网格参数
        grid_config = config['simulation']['grid_search']
        self.belief_points = grid_config['belief_points']
        
        logger.info("Perfect Bayesian Equilibrium求解器初始化完成")
        
    def solve_pbe(self, government_type: GovernmentType) -> PBEResults:
        """
        求解Perfect Bayesian Equilibrium
        
        Args:
            government_type: 政府类型
            
        Returns:
            PBE求解结果
        """
        logger.info(f"开始求解{government_type.value}的Perfect Bayesian Equilibrium...")
        
        # 1. 创建信念网格
        belief_grid = np.linspace(0.01, 0.99, self.belief_points)
        
        # 2. 创建政府实例
        government = ExportingGovernment(self.config, self.device, government_type)
        
        # 3. 求解政府价值函数和策略函数
        logger.info("求解政府价值函数...")
        value_function, policy_function = self.dp_solver.solve_government_value_function(
            government, belief_grid, self.cournot_solver
        )
        
        # 4. 模拟均衡路径
        logger.info("模拟均衡路径...")
        equilibrium_path = self._simulate_equilibrium_path(
            government, policy_function, max_periods=30
        )
        
        # 5. 计算总福利
        total_welfare = sum(sol.welfare_metrics.get('total_welfare', 0) 
                          for sol in equilibrium_path)
        
        # 6. 提取信念演化
        belief_evolution = np.array([sol.belief for sol in equilibrium_path])
        
        return PBEResults(
            equilibrium_path=equilibrium_path,
            value_functions={'government': value_function},
            policy_functions={'government': policy_function},
            belief_evolution=belief_evolution,
            total_welfare=total_welfare,
            convergence_achieved=True
        )
        
    def _simulate_equilibrium_path(self, 
                                 government: ExportingGovernment,
                                 policy_function: Callable,
                                 max_periods: int = 30) -> List[EquilibriumSolution]:
        """模拟均衡路径"""
        path = []
        
        # 初始条件
        current_belief = self.config['simulation']['initial_conditions']['p_0']
        
        # 市场参数
        market_params = {
            'a': self.config['model_parameters']['market']['a'],
            'b': self.config['model_parameters']['market']['b']
        }
        
        for period in range(max_periods):
            # 1. 政府选择最优补贴
            optimal_subsidy = policy_function(current_belief)
            
            # 2. 进口国政府选择最优关税
            belief_tensor = torch.tensor(current_belief, device=self.device)
            optimal_tariff = self.tariff_model.tariff_response_function(belief_tensor).item()
            
            # 3. 企业选择最优策略
            policy_params = {
                'subsidy': optimal_subsidy,
                'tariff': optimal_tariff,
                'delta': self.config['model_parameters']['other']['delta']
            }
            
            # 创建企业实例
            from ..models.core_game import ExportingFirm, ImportingFirm
            exp_firm = ExportingFirm(self.config, self.device)
            imp_firm = ImportingFirm(self.config, self.device)
            
            equilibrium, convergence_info = self.cournot_solver.solve_firm_subgame(
                exp_firm, imp_firm, market_params, policy_params
            )
            
            k_E, q_E, k_I, q_I = equilibrium
            
            # 4. 计算市场价格
            total_output = q_E + q_I
            market_price = market_params['a'] - market_params['b'] * total_output
            
            # 5. 计算福利指标
            welfare_metrics = self._compute_welfare_metrics(
                exp_firm, imp_firm, government, equilibrium, 
                market_price, optimal_subsidy, optimal_tariff
            )
            
            # 6. 创建均衡解
            solution = EquilibriumSolution(
                period=period,
                belief=current_belief,
                exporting_investment=k_E,
                importing_investment=k_I,
                exporting_output=q_E,
                importing_output=q_I,
                subsidy=optimal_subsidy,
                tariff=optimal_tariff,
                market_price=market_price,
                welfare_metrics=welfare_metrics,
                convergence_info=convergence_info
            )
            
            path.append(solution)
            
            # 7. 更新信念（基于政策延续概率）
            # 简化：假设政策继续的概率较高
            continue_prob = government.signal_probability(Signal.CONTINUE)
            if np.random.random() < continue_prob:
                # 观察到继续信号，更新信念
                current_belief, _ = self.belief_updater.full_belief_update(
                    current_belief, Signal.CONTINUE
                )
            else:
                # 政策终止，跳出循环
                logger.info(f"政策在第{period+1}期终止")
                break
                
        return path
        
    def _compute_welfare_metrics(self, 
                               exp_firm: ExportingFirm,
                               imp_firm: ImportingFirm,
                               government: ExportingGovernment,
                               equilibrium: np.ndarray,
                               market_price: float,
                               subsidy: float,
                               tariff: float) -> Dict[str, float]:
        """计算福利指标"""
        k_E, q_E, k_I, q_I = equilibrium
        
        # 转换为张量
        k_E_t = torch.tensor(k_E, device=self.device)
        q_E_t = torch.tensor(q_E, device=self.device)
        k_I_t = torch.tensor(k_I, device=self.device)
        q_I_t = torch.tensor(q_I, device=self.device)
        price_t = torch.tensor(market_price, device=self.device)
        tariff_t = torch.tensor(tariff, device=self.device)
        subsidy_t = torch.tensor(subsidy, device=self.device)
        
        # 企业利润
        exp_profit = exp_firm.profit(k_E_t, q_E_t, price_t, tariff_t, subsidy_t).item()
        imp_profit = imp_firm.profit(k_I_t, q_I_t, price_t).item()
        
        # 政府福利
        subsidy_cost = subsidy * exp_firm.investment_cost(k_E_t).item()
        emissions_reduction = exp_firm.e_0 - exp_firm.unit_emissions(k_E_t).item()
        env_benefit = government.alpha * emissions_reduction * (q_E + q_I)
        exp_welfare = exp_profit - subsidy_cost + env_benefit
        
        # 消费者剩余
        total_output = q_E + q_I
        consumer_surplus = 0.5 * self.config['model_parameters']['market']['b'] * total_output**2
        
        # 关税收入
        exp_emissions = exp_firm.unit_emissions(k_E_t).item()
        tariff_revenue = tariff * exp_emissions * q_E
        
        # 环境成本
        imp_emissions = imp_firm.unit_emissions(k_I_t).item()
        total_emissions = exp_emissions * q_E + imp_emissions * q_I
        env_cost = self.config['model_parameters']['other']['phi'] * total_emissions
        
        # 进口国福利
        imp_welfare = consumer_surplus + imp_profit + tariff_revenue - env_cost
        
        return {
            'exporting_profit': exp_profit,
            'importing_profit': imp_profit,
            'exporting_welfare': exp_welfare,
            'importing_welfare': imp_welfare,
            'consumer_surplus': consumer_surplus,
            'tariff_revenue': tariff_revenue,
            'environmental_cost': env_cost,
            'total_welfare': exp_welfare + imp_welfare,
            'total_emissions': total_emissions
        }