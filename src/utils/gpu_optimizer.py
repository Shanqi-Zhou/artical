"""
GPU加速和CUDA优化工具模块
提供GPU内存管理、批处理优化和设备配置功能
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
import gc
import psutil
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """GPU加速优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化GPU优化器
        
        Args:
            config: GPU配置参数
        """
        self.config = config
        self.device = self._setup_device()
        self.memory_fraction = config.get('memory_fraction', 0.8)
        self.batch_size = config.get('batch_size', 512)
        
        # 配置CUDA优化
        self._configure_cuda()
        
        logger.info(f"GPU优化器初始化完成，设备: {self.device}")
        
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"自动选择CUDA设备: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                logger.warning("CUDA不可用，使用CPU")
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"强制使用CUDA设备: {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("CUDA不可用但配置要求使用CUDA")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU设备")
            
        return device
        
    def _configure_cuda(self):
        """配置CUDA优化设置"""
        if self.device.type == 'cuda':
            # 启用cuDNN benchmark
            if self.config.get('enable_cudnn_benchmark', True):
                torch.backends.cudnn.benchmark = True
                logger.info("已启用cuDNN benchmark优化")
                
            # 设置cuDNN确定性
            if self.config.get('enable_cudnn_deterministic', False):
                torch.backends.cudnn.deterministic = True
                logger.info("已启用cuDNN确定性模式")
                
            # 显示GPU内存信息
            self._log_gpu_memory_info()
            
    def _log_gpu_memory_info(self):
        """记录GPU内存信息"""
        if self.device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            
            logger.info(f"GPU内存状态:")
            logger.info(f"  总内存: {total_memory:.2f} GB")
            logger.info(f"  已分配: {allocated_memory:.2f} GB")
            logger.info(f"  已保留: {reserved_memory:.2f} GB")
            
    def estimate_batch_size(self, tensor_shape: Tuple[int, ...], 
                          dtype: torch.dtype = torch.float32,
                          safety_factor: float = 0.8) -> int:
        """
        估算最优批处理大小
        
        Args:
            tensor_shape: 单个样本的张量形状
            dtype: 数据类型
            safety_factor: 安全系数
            
        Returns:
            建议的批处理大小
        """
        if self.device.type == 'cpu':
            # CPU模式下基于内存估算
            available_memory = psutil.virtual_memory().available / (1024**3)
            bytes_per_element = torch.tensor([], dtype=dtype).element_size()
            elements_per_sample = np.prod(tensor_shape)
            memory_per_sample = elements_per_sample * bytes_per_element / (1024**3)
            
            max_batch_size = int(available_memory * safety_factor / memory_per_sample)
            
        else:
            # GPU模式下基于显存估算
            available_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = available_memory * safety_factor / (1024**3)
            
            bytes_per_element = torch.tensor([], dtype=dtype).element_size()
            elements_per_sample = np.prod(tensor_shape)
            memory_per_sample = elements_per_sample * bytes_per_element / (1024**3)
            
            # 考虑梯度和中间计算的内存开销(约3倍)
            memory_per_sample *= 3
            
            max_batch_size = int(available_memory / memory_per_sample)
            
        # 确保批处理大小在合理范围内
        suggested_batch_size = min(max_batch_size, self.batch_size)
        suggested_batch_size = max(suggested_batch_size, 16)  # 最小批处理大小
        
        logger.info(f"建议批处理大小: {suggested_batch_size}")
        return suggested_batch_size
        
    @contextmanager
    def memory_efficient_context(self):
        """内存高效的上下文管理器"""
        if self.device.type == 'cuda':
            # 记录初始内存状态
            initial_memory = torch.cuda.memory_allocated(0)
            
        try:
            yield
        finally:
            # 清理内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
                
                final_memory = torch.cuda.memory_allocated(0)
                memory_diff = (final_memory - initial_memory) / (1024**2)
                
                if memory_diff > 100:  # 如果内存增长超过100MB
                    logger.warning(f"内存使用增长: {memory_diff:.2f} MB")
                    
    def optimize_tensor(self, tensor: torch.Tensor, 
                       requires_grad: bool = False) -> torch.Tensor:
        """
        优化张量设置
        
        Args:
            tensor: 输入张量
            requires_grad: 是否需要梯度
            
        Returns:
            优化后的张量
        """
        # 移动到目标设备
        tensor = tensor.to(self.device)
        
        # 设置梯度要求
        tensor.requires_grad_(requires_grad)
        
        # 数据类型优化
        precision = self.config.get('precision', 'float32')
        if precision == 'float16' and self.device.type == 'cuda':
            tensor = tensor.half()
        elif precision == 'float64':
            tensor = tensor.double()
        else:
            tensor = tensor.float()
            
        return tensor
        
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU缓存已清理")
            
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计信息"""
        stats = {}
        
        if self.device.type == 'cuda':
            stats['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats['gpu_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
            stats['gpu_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
            stats['gpu_free'] = stats['gpu_total'] - stats['gpu_reserved']
            
        # CPU内存
        memory = psutil.virtual_memory()
        stats['cpu_total'] = memory.total / (1024**3)
        stats['cpu_available'] = memory.available / (1024**3)
        stats['cpu_used'] = memory.used / (1024**3)
        
        return stats

def setup_reproducibility(seed: int = 42):
    """
    设置可重复性
    
    Args:
        seed: 随机种子
    """
    # Python随机种子
    import random
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"随机种子设置为: {seed}")

def batch_process(data: torch.Tensor, 
                 process_fn: callable,
                 batch_size: int,
                 device: torch.device,
                 **kwargs) -> torch.Tensor:
    """
    批处理数据
    
    Args:
        data: 输入数据
        process_fn: 处理函数
        batch_size: 批处理大小
        device: 计算设备
        **kwargs: 额外参数
        
    Returns:
        处理后的数据
    """
    results = []
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size].to(device)
        
        with torch.no_grad():
            batch_result = process_fn(batch, **kwargs)
            
        results.append(batch_result.cpu())
        
        # 清理批处理后的内存
        del batch, batch_result
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    return torch.cat(results, dim=0)

def auto_mixed_precision_context():
    """自动混合精度上下文"""
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    else:
        return torch.no_grad()