"""
通用工具模块
提供配置管理、日志设置、参数跟踪等通用功能
"""

import yaml
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import torch
import pandas as pd

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"无法加载配置文件 {self.config_path}: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键
        
        Args:
            key: 配置键，支持点分隔的嵌套键如 'model.learning_rate'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def update(self, key: str, value: Any, reason: str = ""):
        """
        更新配置值并记录修改
        
        Args:
            key: 配置键
            value: 新值
            reason: 修改原因
        """
        keys = key.split('.')
        config_ref = self.config
        
        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
            
        # 记录旧值
        old_value = config_ref.get(keys[-1], None)
        
        # 更新值
        config_ref[keys[-1]] = value
        
        # 记录修改
        self._track_parameter_change(key, old_value, value, reason)
        
    def _track_parameter_change(self, key: str, old_value: Any, 
                              new_value: Any, reason: str):
        """记录参数修改"""
        if not self.get('parameter_tracking.enable', False):
            return
            
        track_file = self.get('parameter_tracking.track_file', 
                            'logs/parameter_changes.json')
        
        # 创建目录
        os.makedirs(os.path.dirname(track_file), exist_ok=True)
        
        # 加载现有记录
        if os.path.exists(track_file):
            with open(track_file, 'r', encoding='utf-8') as f:
                changes = json.load(f)
        else:
            changes = []
            
        # 添加新记录
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'parameter': key,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'reason': reason,
            'change_magnitude': self._calculate_change_magnitude(old_value, new_value)
        }
        
        changes.append(change_record)
        
        # 保存记录
        with open(track_file, 'w', encoding='utf-8') as f:
            json.dump(changes, f, indent=2, ensure_ascii=False)
            
    def _calculate_change_magnitude(self, old_value: Any, new_value: Any) -> float:
        """计算参数变化幅度"""
        try:
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if old_value == 0:
                    return float('inf') if new_value != 0 else 0
                return abs((new_value - old_value) / old_value)
            else:
                return 1.0 if old_value != new_value else 0.0
        except:
            return 1.0

class Logger:
    """日志管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化日志管理器
        
        Args:
            config: 日志配置
        """
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        log_level = self.config.get('logging.level', 'INFO')
        log_dir = self.config.get('logging.log_dir', 'logs')
        save_logs = self.config.get('logging.save_logs', True)
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 根日志器配置
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器
        if save_logs:
            log_file = os.path.join(log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

class ResultsManager:
    """结果管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化结果管理器
        
        Args:
            config: 结果配置
        """
        self.config = config
        self.results_dir = config.get('output.results_dir', 'results')
        self.data_dir = config.get('output.export.data_dir', 'data')
        
        # 创建目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
    def save_results(self, results: Dict[str, Any], 
                    filename: str, format: str = 'json'):
        """
        保存实验结果
        
        Args:
            results: 结果字典
            filename: 文件名
            format: 保存格式 ('json', 'yaml', 'pkl')
        """
        filepath = os.path.join(self.results_dir, f"{filename}.{format}")
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, 
                         default=self._json_serializer)
        elif format == 'yaml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, 
                         allow_unicode=True)
        elif format == 'pkl':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        logging.info(f"结果已保存到: {filepath}")
        
    def save_data(self, data: np.ndarray, name: str, 
                  metadata: Optional[Dict[str, Any]] = None):
        """
        保存实验数据
        
        Args:
            data: 数据数组
            name: 数据名称
            metadata: 元数据
        """
        data_format = self.config.get('output.export.data_format', 'h5')
        
        if data_format == 'h5':
            import h5py
            filepath = os.path.join(self.data_dir, f"{name}.h5")
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=data)
                if metadata:
                    for key, value in metadata.items():
                        f.attrs[key] = value
                        
        elif data_format == 'csv':
            filepath = os.path.join(self.data_dir, f"{name}.csv")
            if data.ndim == 1:
                pd.Series(data).to_csv(filepath, index=False)
            elif data.ndim == 2:
                pd.DataFrame(data).to_csv(filepath, index=False)
            else:
                raise ValueError("CSV格式仅支持1D或2D数据")
                
        elif data_format == 'pkl':
            import pickle
            filepath = os.path.join(self.data_dir, f"{name}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump({'data': data, 'metadata': metadata}, f)
                
        logging.info(f"数据已保存到: {filepath}")
        
    def _json_serializer(self, obj):
        """JSON序列化器，处理NumPy和PyTorch类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            raise TypeError(f"无法序列化类型: {type(obj)}")

def create_experiment_id() -> str:
    """创建实验ID"""
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def ensure_dir(directory: str):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)

def format_number(value: float, precision: int = 4) -> str:
    """格式化数字输出"""
    if abs(value) < 1e-10:
        return "0.0000"
    elif abs(value) >= 1e6:
        return f"{value:.2e}"
    else:
        return f"{value:.{precision}f}"

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    验证配置文件
    
    Args:
        config: 配置字典
        
    Returns:
        错误列表
    """
    errors = []
    
    # 必需的配置项
    required_keys = [
        'model_parameters.market.a',
        'model_parameters.market.b',
        'model_parameters.government_types.theta_L',
        'model_parameters.government_types.theta_H',
        'simulation.monte_carlo.num_runs',
        'simulation.monte_carlo.periods'
    ]
    
    for key in required_keys:
        keys = key.split('.')
        value = config
        try:
            for k in keys:
                value = value[k]
        except (KeyError, TypeError):
            errors.append(f"缺少必需配置项: {key}")
            
    # 数值范围验证
    if 'model_parameters' in config:
        params = config['model_parameters']
        
        # 验证概率参数
        if 'government_types' in params:
            gt = params['government_types']
            if 'theta_L' in gt and not (0 <= gt['theta_L'] <= 1):
                errors.append("theta_L必须在[0,1]范围内")
            if 'theta_H' in gt and not (0 <= gt['theta_H'] <= 1):
                errors.append("theta_H必须在[0,1]范围内")
            if 'theta_L' in gt and 'theta_H' in gt:
                if gt['theta_L'] >= gt['theta_H']:
                    errors.append("theta_L必须小于theta_H")
                    
        # 验证行为摩擦参数
        if 'behavioral_frictions' in params:
            bf = params['behavioral_frictions']
            if 'epsilon' in bf and not (0 <= bf['epsilon'] < 0.5):
                errors.append("epsilon必须在[0,0.5)范围内")
            if 'eta' in bf and not (0 < bf['eta'] <= 1):
                errors.append("eta必须在(0,1]范围内")
                
    return errors