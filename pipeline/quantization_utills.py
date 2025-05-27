# pipeline/quantization_utils.py
import torch
import torch.quantization
import logging
from torch.utils.data import DataLoader, Subset
from typing import Optional, Union, Dict, Any
import numpy as np
import pickle

logger = logging.getLogger(__name__)


def get_calibration_dataloader(dataset, tokenizer, data_collator, batch_size, num_batches):
    """为静态量化准备校准数据加载器"""
    if dataset is None or len(dataset) == 0:
        logger.warning("Warning: Calibration dataset is empty.")
        return None

    num_samples_to_take = min(batch_size * num_batches, len(dataset))
    if num_samples_to_take == 0 and len(dataset) > 0:
        num_samples_to_take = len(dataset)

    if num_samples_to_take == 0:
        logger.warning("Warning: No samples selected for calibration.")
        return None

    indices = torch.randperm(len(dataset))[:num_samples_to_take].tolist()
    calibration_subset = Subset(dataset, indices)
    return DataLoader(calibration_subset, batch_size=batch_size, collate_fn=data_collator)


def quantize_model_dynamic(model, backend='fbgemm'):
    """应用动态量化到模型"""
    logger.info(f"Applying dynamic quantization with backend: {backend}")

    # 确保模型处于评估模式
    model.eval()

    # 指定要量化的层类型
    modules_to_quantize = {torch.nn.Linear}

    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec=modules_to_quantize,
            dtype=torch.qint8,
            backend=backend
        )
        logger.info("Dynamic quantization completed successfully")
        return quantized_model
    except Exception as e:
        logger.error(f"Error during dynamic quantization: {e}")
        # 如果量化失败，返回原始模型
        return model


def quantize_model_static(model, dataloader, backend='fbgemm'):
    """应用静态量化到模型"""
    logger.info(f"Applying static quantization with backend: {backend}")

    model.eval()

    # 设置量化配置
    model.qconfig = torch.quantization.get_default_qconfig(backend)

    # 准备量化
    torch.quantization.prepare(model, inplace=True)

    # 校准步骤
    if dataloader:
        logger.info("Calibrating model for static quantization...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                try:
                    # 移动批次到模型的设备
                    device = next(model.parameters()).device
                    inputs = {}
                    for k, v in batch.items():
                        if k in ['input_ids', 'attention_mask', 'token_type_ids'] and v is not None:
                            inputs[k] = v.to(device)

                    if not inputs:
                        continue

                    # 前向传播进行校准
                    model(**inputs)

                    if i >= 10:  # 限制校准批次数量
                        break

                except Exception as e:
                    logger.warning(f"Warning: Error during calibration step {i}: {e}")
                    continue

        logger.info("Calibration finished.")
    else:
        logger.warning("Warning: No dataloader provided for static quantization calibration.")

    # 转换为量化模型
    try:
        torch.quantization.convert(model, inplace=True)
        logger.info("Static quantization completed successfully")
    except Exception as e:
        logger.error(f"Error during static quantization conversion: {e}")

    return model


def simulate_quantization_communication_cost(state_dict, dtype=torch.qint8):
    """模拟量化后的通信成本"""
    total_size = 0
    float_size = 0

    for key, param in state_dict.items():
        if param.is_floating_point():
            float_size += param.numel() * 4  # float32 = 4 bytes
            if dtype == torch.qint8:
                total_size += param.numel() * 1  # int8 = 1 byte
            elif dtype == torch.qint16:
                total_size += param.numel() * 2  # int16 = 2 bytes
        else:
            # 非浮点参数保持原样
            total_size += param.numel() * param.element_size()
            float_size += param.numel() * param.element_size()

    compression_ratio = float_size / total_size if total_size > 0 else 1.0

    logger.info(f"Communication cost simulation - Original: {float_size / 1024 / 1024:.2f}MB, "
                f"Quantized: {total_size / 1024 / 1024:.2f}MB, "
                f"Compression ratio: {compression_ratio:.2f}x")

    return {
        'original_size_mb': float_size / 1024 / 1024,
        'quantized_size_mb': total_size / 1024 / 1024,
        'compression_ratio': compression_ratio
    }


def dequantize_model_weights(quantized_model):
    """从量化模型中提取浮点权重用于聚合"""
    try:
        # 创建一个新的状态字典来存储去量化的权重
        dequantized_state_dict = {}

        for name, param in quantized_model.named_parameters():
            if hasattr(param, 'dequantize'):
                # 如果参数是量化的，则去量化
                dequantized_state_dict[name] = param.dequantize()
            else:
                # 如果参数不是量化的，直接复制
                dequantized_state_dict[name] = param.clone()

        # 处理缓冲区
        for name, buffer in quantized_model.named_buffers():
            if hasattr(buffer, 'dequantize'):
                dequantized_state_dict[name] = buffer.dequantize()
            else:
                dequantized_state_dict[name] = buffer.clone()

        logger.info("Successfully dequantized model weights")
        return dequantized_state_dict

    except Exception as e:
        logger.error(f"Error during dequantization: {e}")
        # 如果去量化失败，返回原始状态字典
        return quantized_model.state_dict()


def get_model_size_info(model):
    """获取模型大小信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 估算模型大小（假设float32）
    model_size_mb = total_params * 4 / 1024 / 1024

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }


# 🔧 新增：缺失的实际量化函数
'''def quantize_state_dict_real(state_dict, config):
    """
    对状态字典进行真实量化
    """
    try:
        quantized_dict = {}
        total_params = 0
        quantized_params = 0
        original_size = 0
        quantized_size = 0
        
        for key, tensor in state_dict.items():
            total_params += tensor.numel()
            original_size += tensor.numel() * tensor.element_size()
            
            if tensor.is_floating_point() and tensor.numel() > 100:  # 只量化较大的浮点张量
                # 执行8位量化
        # 计算量化参数
                tensor_min = tensor.min().item()
                tensor_max = tensor.max().item()
        
        # 避免零范围
        if abs(tensor_max - tensor_min) < 1e-8:
            scale = 1.0
            zero_point = 0
        else:
            scale = (tensor_max - tensor_min) / 255.0
            zero_point = int(-tensor_min / scale)
            zero_point = max(0, min(255, zero_point))
        
        # 量化
        quantized_tensor = tensor / scale + zero_point
        quantized_data = torch.clamp(quantized_tensor, min=0, max=255).round().to(torch.uint8)
                
        quantized_dict[key] = {
            'quantized': True,
            'data': quantized_data,
            'scale': scale,
            'zero_point': zero_point,
            'original_shape': tensor.shape,
            'original_dtype': tensor.dtype
                }
                
                quantized_params += tensor.numel()
                quantized_size += quantized_data.numel() * 1 + 8  # 数据 + scale + zero_point
            else:
                # 不量化的张量直接保存
                quantized_dict[key] = tensor.clone()
                quantized_size += tensor.numel() * tensor.element_size()
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        
        stats = {
            'total_params': total_params,
            'quantized_params': quantized_params,
            'original_size_mb': original_size / 1024 / 1024,
            'quantized_size_mb': quantized_size / 1024 / 1024,
            'compression_ratio': compression_ratio
        }
        
        return quantized_dict, stats
        
    except Exception as e:
        logger.error(f"Real quantization failed: {e}")
        # 返回原始状态字典和默认统计信息
        original_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        stats = {
            'total_params': sum(t.numel() for t in state_dict.values()),
            'quantized_params': 0,
            'original_size_mb': original_size / 1024 / 1024,
            'quantized_size_mb': original_size / 1024 / 1024,
            'compression_ratio': 1.0
        }
        return state_dict, stats'''

def quantize_state_dict_real(state_dict, config):
    """
    对状态字典进行真实量化
    """
    try:
        quantized_dict = {}
        total_params = 0
        quantized_params = 0
        original_size = 0
        quantized_size = 0
        
        for key, tensor in state_dict.items():
            total_params += tensor.numel()
            original_size += tensor.numel() * tensor.element_size()
            
            if tensor.is_floating_point() and tensor.numel() > 100:  # 只量化较大的浮点张量
                # 执行8位量化
                # 计算量化参数
                tensor_min = tensor.min().item()
                tensor_max = tensor.max().item()
        
                # 避免零范围
                if abs(tensor_max - tensor_min) < 1e-8:
                    scale = 1.0
                    zero_point = 0
                else:
                    scale = (tensor_max - tensor_min) / 255.0
                    zero_point = int(-tensor_min / scale)
                    zero_point = max(0, min(255, zero_point))
        
                # 量化
                quantized_tensor = tensor / scale + zero_point
                quantized_data = torch.clamp(quantized_tensor, min=0, max=255).round().to(torch.uint8)
                
                quantized_dict[key] = {
                    'quantized': True,
                    'data': quantized_data,
                    'scale': scale,
                    'zero_point': zero_point,
                    'original_shape': tensor.shape,
                    'original_dtype': tensor.dtype
                }
                
                quantized_params += tensor.numel()
                quantized_size += quantized_data.numel() * 1 + 8  # 数据 + scale + zero_point
            else:
                # 不量化的张量直接保存
                quantized_dict[key] = tensor.clone()
                quantized_size += tensor.numel() * tensor.element_size()
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        
        stats = {
            'total_params': total_params,
            'quantized_params': quantized_params,
            'original_size_mb': original_size / 1024 / 1024,
            'quantized_size_mb': quantized_size / 1024 / 1024,
            'compression_ratio': compression_ratio
        }
        
        return quantized_dict, stats
        
    except Exception as e:
        logger.error(f"Real quantization failed: {e}")
        # 返回原始状态字典和默认统计信息
        original_size = sum(t.numel() * t.element_size() for t in state_dict.values())
        stats = {
            'total_params': sum(t.numel() for t in state_dict.values()),
            'quantized_params': 0,
            'original_size_mb': original_size / 1024 / 1024,
            'quantized_size_mb': original_size / 1024 / 1024,
            'compression_ratio': 1.0
        }
        return state_dict, stats


def dequantize_state_dict_real(quantized_dict):
    """
    将量化的状态字典去量化
    """
    try:
        dequantized_dict = {}
        
        for key, value in quantized_dict.items():
            if isinstance(value, dict) and value.get('quantized', False):
                # 去量化
                quantized_data = value['data']
                scale = value['scale']
                zero_point = value['zero_point']
                original_shape = value['original_shape']
                original_dtype = value['original_dtype']
                
                # 去量化公式: (quantized_data - zero_point) * scale
                dequantized = (quantized_data.float() - zero_point) * scale
                dequantized = dequantized.to(original_dtype).reshape(original_shape)
                
                dequantized_dict[key] = dequantized
            else:
                # 非量化张量直接复制
                dequantized_dict[key] = value.clone() if hasattr(value, 'clone') else value
        
        return dequantized_dict
        
    except Exception as e:
        logger.error(f"Real dequantization failed: {e}")
        return quantized_dict


def serialize_quantized_weights(quantized_dict):
    """
    序列化量化权重用于传输
    """
    try:
        return pickle.dumps(quantized_dict)
    except Exception as e:
        logger.error(f"Serialization failed: {e}")
        return None


def deserialize_quantized_weights(serialized_data):
    """
    反序列化量化权重
    """
    try:
        return pickle.loads(serialized_data)
    except Exception as e:
        logger.error(f"Deserialization failed: {e}")
        return None


def measure_real_communication_savings(original_dict, quantized_dict):
    """
    测量实际的通信节省
    """
    try:
        # 计算原始大小
        original_size = 0
        for tensor in original_dict.values():
            original_size += tensor.numel() * tensor.element_size()
        
        # 计算量化后大小
        quantized_size = 0
        for key, value in quantized_dict.items():
            if isinstance(value, dict) and value.get('quantized', False):
                quantized_size += value['data'].numel() * 1 + 8  # 数据 + 元数据
            else:
                quantized_size += value.numel() * value.element_size()
        
        savings_percentage = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        
        return {
            'original_size_bytes': original_size,
            'quantized_size_bytes': quantized_size,
            'savings_percentage': savings_percentage,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1.0
        }
        
    except Exception as e:
        logger.error(f"Communication savings measurement failed: {e}")
        return {
            'original_size_bytes': 0,
            'quantized_size_bytes': 0,
            'savings_percentage': 0,
            'compression_ratio': 1.0
        }


class QuantizationConfig:
    """量化配置类"""

    def __init__(self,
                 apply_quantization: bool = False,
                 quantization_type: str = "dynamic",
                 quantization_backend: str = "fbgemm",
                 calibration_batch_size: int = 8,
                 num_calibration_batches: int = 10,
                 quantize_global_model: bool = False):
        self.apply_quantization = apply_quantization
        self.quantization_type = quantization_type
        self.quantization_backend = quantization_backend
        self.calibration_batch_size = calibration_batch_size
        self.num_calibration_batches = num_calibration_batches
        self.quantize_global_model = quantize_global_model

    def validate(self):
        """验证配置"""
        if self.quantization_type not in ["dynamic", "static"]:
            raise ValueError(f"Invalid quantization_type: {self.quantization_type}")

        if self.quantization_backend not in ["fbgemm", "qnnpack"]:
            raise ValueError(f"Invalid quantization_backend: {self.quantization_backend}")

        if self.calibration_batch_size <= 0:
            raise ValueError("calibration_batch_size must be positive")

        if self.num_calibration_batches <= 0:
            raise ValueError("num_calibration_batches must be positive")
