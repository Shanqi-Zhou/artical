"""
蒙特卡洛模拟框架
实现30期动态博弈模拟和多次独立运行，复现论文中的核心实验
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .core_game import GovernmentType, Signal, GameState
from .belief_update import BeliefUpdater, BeliefDrivenTariffModel
from ..solvers.firm_subgame import ImprovedCournotSolver, FDIDecisionModel
from ..utils.gpu_optimizer import GPUOptimizer

logger = logging.getLogger(__name__)

@dataclass
class SimulationState:
    """单期模拟状态"""
    period: int
    belief: float
    subsidy: float
    tariff: float
    k_E: float
    q_E: float
    k_I: float
    q_I: float
    market_price: float
    profit_E: float
    profit_I: float
    welfare_E: float
    welfare_I: float
    total_emissions: float
    signal_realized: str
    policy_terminated: bool

@dataclass
class SimulationRun:
    """单次模拟运行结果"""
    run_id: int
    government_type: str
    epsilon: float
    eta: float
    num_periods: int
    states: List[SimulationState]
    final_belief: float
    termination_period: Optional[int]
    total_welfare: float
    avg_emissions: float
    convergence_info: Dict[str, Any]

@dataclass
class MonteCarloResults:
    """蒙特卡洛模拟结果"""
    experiment_name: str
    government_type: str
    num_runs: int
    num_periods: int
    epsilon: float
    eta: float
    runs: List[SimulationRun]
    aggregate_stats: Dict[str, Any]
    execution_time: float

class DynamicGameSimulator:
    """动态博弈模拟器"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化动态博弈模拟器
        
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
        self.fdi_model = FDIDecisionModel(config, device)
        
        # 政府类型参数
        gov_types = config['model_parameters']['government_types']
        self.theta_L = gov_types['theta_L']
        self.theta_H = gov_types['theta_H']
        
        # 模拟参数
        sim_config = config['simulation']
        self.initial_belief = sim_config['initial_conditions']['p_0']
        self.initial_subsidy = sim_config['initial_conditions']['s_0']
        
        # 其他参数
        self.alpha = config['model_parameters']['other']['alpha']
        self.phi = config['model_parameters']['other']['phi']
        
        logger.info("动态博弈模拟器初始化完成")
        
    def simulate_single_period(self, 
                              current_state: SimulationState,
                              government_type: GovernmentType,
                              epsilon: float,
                              eta: float) -> Tuple[SimulationState, bool]:
        """
        模拟单期动态博弈
        
        Args:
            current_state: 当前状态
            government_type: 政府类型
            epsilon: 认知偏误参数
            eta: 声誉衰减因子
            
        Returns:
            (新状态, 是否终止)
        """
        # 1. 出口国政府选择补贴强度
        # 这里使用简化的策略：基于信念的启发式策略
        if government_type == GovernmentType.HIGH_COMMITMENT:
            # 高承诺类型更愿意维持高补贴
            base_subsidy = 0.4
            belief_adjustment = (1 - current_state.belief) * 0.2
        else:
            # 低承诺类型的补贴更谨慎
            base_subsidy = 0.2
            belief_adjustment = (1 - current_state.belief) * 0.1
            
        optimal_subsidy = min(base_subsidy + belief_adjustment, 1.0)
        
        # 2. 进口国政府选择关税
        belief_tensor = torch.tensor(current_state.belief, device=self.device)
        optimal_tariff = self.tariff_model.tariff_response_function(belief_tensor).item()
        
        # 3. 企业选择投资和产量
        equilibrium = self.cournot_solver.solve_equilibrium(optimal_subsidy, optimal_tariff)
        
        # 4. 计算福利指标
        welfare_metrics = self._compute_welfare_metrics(
            equilibrium, optimal_subsidy, optimal_tariff
        )
        
        # 5. 生成政策延续信号
        true_theta = self.theta_L if government_type == GovernmentType.HIGH_COMMITMENT else self.theta_H
        signal_realized, policy_terminated = self._generate_signal(true_theta)
        
        # 6. 更新信念（如果政策未终止）
        if not policy_terminated:
            new_belief, _ = self.belief_updater.full_belief_update(
                current_state.belief, Signal.CONTINUE, epsilon, eta
            )
        else:
            new_belief = current_state.belief  # 终止时信念不再更新
            
        # 7. 创建新状态
        new_state = SimulationState(
            period=current_state.period + 1,
            belief=new_belief,
            subsidy=optimal_subsidy,
            tariff=optimal_tariff,
            k_E=equilibrium.exporting_investment,
            q_E=equilibrium.exporting_output,
            k_I=equilibrium.importing_investment,
            q_I=equilibrium.importing_output,
            market_price=equilibrium.market_price,
            profit_E=equilibrium.exporting_profit,
            profit_I=equilibrium.importing_profit,
            welfare_E=welfare_metrics['welfare_E'],
            welfare_I=welfare_metrics['welfare_I'],
            total_emissions=welfare_metrics['total_emissions'],
            signal_realized=signal_realized.value,
            policy_terminated=policy_terminated
        )
        
        return new_state, policy_terminated
        
    def simulate_single_run(self,
                          run_id: int,
                          government_type: GovernmentType,
                          num_periods: int = 30,
                          epsilon: Optional[float] = None,
                          eta: Optional[float] = None,
                          random_seed: Optional[int] = None) -> SimulationRun:
        """
        模拟单次完整运行
        
        Args:
            run_id: 运行ID
            government_type: 政府类型
            num_periods: 模拟期数
            epsilon: 认知偏误参数
            eta: 声誉衰减因子
            random_seed: 随机种子
            
        Returns:
            模拟运行结果
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if epsilon is None:
            epsilon = self.config['model_parameters']['behavioral_frictions']['epsilon']
        if eta is None:
            eta = self.config['model_parameters']['behavioral_frictions']['eta']
            
        # 初始状态
        initial_state = SimulationState(
            period=0,
            belief=self.initial_belief,
            subsidy=self.initial_subsidy,
            tariff=0.2,  # 初始关税
            k_E=0.0, q_E=0.0, k_I=0.0, q_I=0.0,
            market_price=0.0,
            profit_E=0.0, profit_I=0.0,
            welfare_E=0.0, welfare_I=0.0,
            total_emissions=0.0,
            signal_realized="initial",
            policy_terminated=False
        )
        
        states = [initial_state]
        current_state = initial_state
        termination_period = None
        
        # 模拟每期
        for period in range(num_periods):
            try:
                new_state, terminated = self.simulate_single_period(
                    current_state, government_type, epsilon, eta
                )
                states.append(new_state)
                current_state = new_state
                
                if terminated:
                    termination_period = period + 1
                    break
                    
            except Exception as e:
                logger.warning(f"运行{run_id}在第{period+1}期出错: {e}")
                # 使用上一期状态作为终止状态
                termination_period = period + 1
                break
                
        # 计算汇总指标
        final_belief = current_state.belief
        total_welfare = sum(state.welfare_E + state.welfare_I for state in states[1:])
        avg_emissions = np.mean([state.total_emissions for state in states[1:]])
        
        convergence_info = {
            'completed_periods': len(states) - 1,
            'natural_termination': termination_period is not None,
            'final_belief': final_belief
        }
        
        return SimulationRun(
            run_id=run_id,
            government_type=government_type.value,
            epsilon=epsilon,
            eta=eta,
            num_periods=len(states) - 1,
            states=states,
            final_belief=final_belief,
            termination_period=termination_period,
            total_welfare=total_welfare,
            avg_emissions=avg_emissions,
            convergence_info=convergence_info
        )
        
    def _compute_welfare_metrics(self, equilibrium, subsidy: float, tariff: float) -> Dict[str, float]:
        """计算福利指标"""
        # 企业排放
        exp_emissions = self.cournot_solver.unit_emissions_exp(equilibrium.exporting_investment)
        imp_emissions = self.cournot_solver.unit_emissions_imp(equilibrium.importing_investment)
        
        total_emissions = (exp_emissions * equilibrium.exporting_output + 
                          imp_emissions * equilibrium.importing_output)
        
        # 出口国福利
        subsidy_cost = subsidy * 0.5 * self.cournot_solver.exp_gamma * equilibrium.exporting_investment**2
        emissions_reduction = self.cournot_solver.exp_e0 - exp_emissions
        env_benefit = self.alpha * emissions_reduction * (equilibrium.exporting_output + equilibrium.importing_output)
        welfare_E = equilibrium.exporting_profit - subsidy_cost + env_benefit
        
        # 进口国福利  
        total_output = equilibrium.exporting_output + equilibrium.importing_output
        consumer_surplus = 0.5 * self.cournot_solver.b * total_output**2
        tariff_revenue = tariff * exp_emissions * equilibrium.exporting_output
        env_cost = self.phi * total_emissions
        welfare_I = consumer_surplus + equilibrium.importing_profit + tariff_revenue - env_cost
        
        return {
            'welfare_E': welfare_E,
            'welfare_I': welfare_I,
            'total_emissions': total_emissions,
            'consumer_surplus': consumer_surplus,
            'tariff_revenue': tariff_revenue,
            'env_cost': env_cost
        }
        
    def _generate_signal(self, theta: float) -> Tuple[Signal, bool]:
        """生成政策延续信号"""
        if np.random.random() < theta:
            # 政策终止
            return Signal.TERMINATE, True
        else:
            # 政策继续
            return Signal.CONTINUE, False

class MonteCarloSimulator:
    """蒙特卡洛模拟器"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化蒙特卡洛模拟器
        
        Args:
            config: 配置参数
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.game_simulator = DynamicGameSimulator(config, device)
        
        # GPU优化器
        self.gpu_optimizer = GPUOptimizer(config.get('gpu', {}))
        
        logger.info("蒙特卡洛模拟器初始化完成")
        
    def run_monte_carlo(self,
                       government_type: GovernmentType,
                       num_runs: int = 20,
                       num_periods: int = 30,
                       epsilon: Optional[float] = None,
                       eta: Optional[float] = None,
                       parallel: bool = False,
                       experiment_name: str = "baseline") -> MonteCarloResults:
        """
        运行蒙特卡洛模拟
        
        Args:
            government_type: 政府类型
            num_runs: 运行次数
            num_periods: 每次运行的期数
            epsilon: 认知偏误参数
            eta: 声誉衰减因子
            parallel: 是否并行计算
            experiment_name: 实验名称
            
        Returns:
            蒙特卡洛结果
        """
        start_time = time.time()
        
        logger.info(f"开始{experiment_name}实验:")
        logger.info(f"  政府类型: {government_type.value}")
        logger.info(f"  运行次数: {num_runs}")
        logger.info(f"  模拟期数: {num_periods}")
        logger.info(f"  认知偏误: {epsilon}")
        logger.info(f"  声誉衰减: {eta}")
        
        if epsilon is None:
            epsilon = self.config['model_parameters']['behavioral_frictions']['epsilon']
        if eta is None:
            eta = self.config['model_parameters']['behavioral_frictions']['eta']
            
        # 执行模拟
        if parallel and num_runs > 4:
            runs = self._run_parallel_simulation(
                government_type, num_runs, num_periods, epsilon, eta
            )
        else:
            runs = self._run_sequential_simulation(
                government_type, num_runs, num_periods, epsilon, eta
            )
            
        # 计算汇总统计
        aggregate_stats = self._compute_aggregate_statistics(runs)
        
        execution_time = time.time() - start_time
        
        logger.info(f"模拟完成，耗时: {execution_time:.2f}秒")
        logger.info(f"平均最终信念: {aggregate_stats['final_belief_mean']:.3f}")
        logger.info(f"政策终止率: {aggregate_stats['termination_rate']:.1%}")
        
        return MonteCarloResults(
            experiment_name=experiment_name,
            government_type=government_type.value,
            num_runs=num_runs,
            num_periods=num_periods,
            epsilon=epsilon,
            eta=eta,
            runs=runs,
            aggregate_stats=aggregate_stats,
            execution_time=execution_time
        )
        
    def _run_sequential_simulation(self,
                                 government_type: GovernmentType,
                                 num_runs: int,
                                 num_periods: int,
                                 epsilon: float,
                                 eta: float) -> List[SimulationRun]:
        """顺序运行模拟"""
        runs = []
        
        with self.gpu_optimizer.memory_efficient_context():
            for run_id in tqdm(range(num_runs), desc="Monte Carlo模拟"):
                random_seed = self.config['experiment']['random_seed'] + run_id
                
                run = self.game_simulator.simulate_single_run(
                    run_id, government_type, num_periods, epsilon, eta, random_seed
                )
                runs.append(run)
                
                # 定期清理GPU内存
                if (run_id + 1) % 5 == 0:
                    self.gpu_optimizer.clear_cache()
                    
        return runs
        
    def _run_parallel_simulation(self,
                               government_type: GovernmentType,
                               num_runs: int,
                               num_periods: int,
                               epsilon: float,
                               eta: float) -> List[SimulationRun]:
        """并行运行模拟（暂时不实现，避免复杂性）"""
        logger.info("并行模拟暂不支持，使用顺序模拟")
        return self._run_sequential_simulation(
            government_type, num_runs, num_periods, epsilon, eta
        )
        
    def _compute_aggregate_statistics(self, runs: List[SimulationRun]) -> Dict[str, Any]:
        """计算汇总统计"""
        if not runs:
            return {}
            
        # 基本统计
        final_beliefs = [run.final_belief for run in runs]
        termination_periods = [run.termination_period for run in runs if run.termination_period is not None]
        total_welfares = [run.total_welfare for run in runs]
        avg_emissions = [run.avg_emissions for run in runs]
        
        # 信念演化统计
        max_periods = max(len(run.states) - 1 for run in runs)
        belief_trajectories = []
        
        for run in runs:
            trajectory = [state.belief for state in run.states[1:]]  # 跳过初始状态
            # 用最后一个值填充到最大长度
            while len(trajectory) < max_periods:
                trajectory.append(trajectory[-1] if trajectory else 0.5)
            belief_trajectories.append(trajectory[:max_periods])
            
        belief_trajectories = np.array(belief_trajectories)
        
        return {
            'final_belief_mean': np.mean(final_beliefs),
            'final_belief_std': np.std(final_beliefs),
            'final_belief_median': np.median(final_beliefs),
            'termination_rate': len(termination_periods) / len(runs),
            'avg_termination_period': np.mean(termination_periods) if termination_periods else None,
            'total_welfare_mean': np.mean(total_welfares),
            'total_welfare_std': np.std(total_welfares),
            'avg_emissions_mean': np.mean(avg_emissions),
            'avg_emissions_std': np.std(avg_emissions),
            'belief_trajectory_mean': np.mean(belief_trajectories, axis=0),
            'belief_trajectory_std': np.std(belief_trajectories, axis=0),
            'belief_trajectory_q25': np.percentile(belief_trajectories, 25, axis=0),
            'belief_trajectory_q75': np.percentile(belief_trajectories, 75, axis=0),
            'num_successful_runs': len(runs)
        }

def run_paper_reproduction_experiments(config: Dict[str, Any], device: torch.device) -> Dict[str, MonteCarloResults]:
    """
    运行论文复现实验
    
    Args:
        config: 配置参数
        device: 计算设备
        
    Returns:
        所有实验结果
    """
    logger.info("开始论文复现实验...")
    
    mc_simulator = MonteCarloSimulator(config, device)
    results = {}
    
    # 基础参数
    num_runs = config['simulation']['monte_carlo']['num_runs']
    num_periods = config['simulation']['monte_carlo']['periods']
    
    # 实验1: 基线实验（高承诺类型）
    logger.info("\n=== 实验1: 基线实验（高承诺类型） ===")
    results['baseline_high_commitment'] = mc_simulator.run_monte_carlo(
        government_type=GovernmentType.HIGH_COMMITMENT,
        num_runs=num_runs,
        num_periods=num_periods,
        experiment_name="baseline_high_commitment"
    )
    
    # 实验2: 基线实验（低承诺类型）
    logger.info("\n=== 实验2: 基线实验（低承诺类型） ===")
    results['baseline_low_commitment'] = mc_simulator.run_monte_carlo(
        government_type=GovernmentType.LOW_COMMITMENT,
        num_runs=num_runs,
        num_periods=num_periods,
        experiment_name="baseline_low_commitment"
    )
    
    # 实验3: 无认知偏误（ε=0）
    logger.info("\n=== 实验3: 无认知偏误实验 ===")
    results['no_cognitive_bias'] = mc_simulator.run_monte_carlo(
        government_type=GovernmentType.HIGH_COMMITMENT,
        num_runs=num_runs,
        num_periods=num_periods,
        epsilon=0.0,
        experiment_name="no_cognitive_bias"
    )
    
    # 实验4: 无声誉衰减（η=1）
    logger.info("\n=== 实验4: 无声誉衰减实验 ===")
    results['no_reputation_decay'] = mc_simulator.run_monte_carlo(
        government_type=GovernmentType.HIGH_COMMITMENT,
        num_runs=num_runs,
        num_periods=num_periods,
        eta=1.0,
        experiment_name="no_reputation_decay"
    )
    
    logger.info("\n论文复现实验完成!")
    return results