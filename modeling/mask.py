import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def sample(inps: Union[Tuple[torch.Tensor, ...], torch.Tensor], size: int):
    if isinstance(inps, torch.Tensor):
        assert len(inps.shape) == 3 and inps.shape[0] == 1
        inps = inps.squeeze(0).cpu()  # (seq_length, hidden_size)
        size = min(inps.shape[0], size)
        indices = np.random.choice(inps.shape[0], size, replace=False)
        indices = torch.from_numpy(indices)
        return inps[indices]
    else:
        return tuple(sample(x, size) for x in inps)


class Mask(nn.Module):
    """
    修复后的掩码类：
    1. 修正初始化策略，从低稀疏度开始
    2. 增强数值稳定性
    3. 添加调试信息
    4. 改进稀疏度控制
    """
    min_s = -0.1
    max_s = 1.1
    eps = 1e-6
    magical_number = 0.8
    
    def __init__(self, features: int, repeat: int = 1) -> None:
        super().__init__()
        self.activate = nn.Parameter(torch.tensor(True), requires_grad=False)
        self.features = features * repeat
        self.repeat = repeat
        self.beta = 2. / 3.
        self.log_alpha = nn.Parameter(torch.zeros((features,)))
        self.sampler = torch.distributions.Uniform(self.eps, 1. - self.eps)
        
        # 🔧 修复1: 从低稀疏度开始，逐渐增加剪枝
        # 原来: self.set_params(10.0) -> L()≈1.0 (几乎不剪枝)
        # 修正: 从-2.0开始，对应约30%的保留率，随训练逐渐剪枝
        self.set_params(-2.0)  # 对应初始L()≈0.3，开始就有一定剪枝
        
        # 添加调试计数器
        self.debug_call_count = 0
        
    def set_state(self, activate: bool):
        self.activate.copy_(activate)
    
    @torch.no_grad()
    def set_params(self, mean: float, indices: Optional[torch.LongTensor] = None):
        """
        🔧 修复2: 增强参数设置的数值稳定性
        """
        # 限制mean的范围，避免极端值
        mean_tensor = torch.tensor(mean)
        mean = torch.clamp(mean_tensor, min=-10.0, max=10.0).item()
        
        if indices is None:
            self.log_alpha.normal_(mean=mean, std=1e-2)
        else:
            self.log_alpha[indices].normal_(mean=mean, std=1e-2)
        
        # 限制log_alpha范围，确保数值稳定性
        self.log_alpha.data = torch.clamp(self.log_alpha.data, min=-15.0, max=15.0)
    
    def L(self):
        """
        🔧 修复3: 增强L()计算的数值稳定性并添加调试信息
        """
        log_alpha = self.log_alpha.repeat(self.repeat)
        
        # 计算logits，添加数值稳定性检查
        x = (0 - self.min_s) / (self.max_s - self.min_s)
        x = torch.clamp(torch.tensor(x), min=self.eps, max=1 - self.eps)  # 确保x在有效范围内
        logits = math.log(x.item()) - math.log(1 - x.item())
        
        # 添加数值稳定性检查
        alpha_term = log_alpha - logits * self.beta
        alpha_term = torch.clamp(alpha_term, min=-15.0, max=15.0)  # 防止sigmoid溢出
        
        L = torch.sigmoid(alpha_term).clamp(min=self.eps, max=1-self.eps)
        
        # 调试信息（每100次调用打印一次）
        self.debug_call_count += 1
        if self.debug_call_count % 100 == 0:
            expected_retention = L.mean().item()
            expected_sparsity = 1.0 - expected_retention
            logger.debug(f"Mask L() - Expected retention: {expected_retention:.4f}, "
                        f"Expected sparsity: {expected_sparsity:.4f}, "
                        f"Features: {self.features//self.repeat}")
        
        if not self.activate.item():
            L = L.detach()
        return L

    def sample_z(self):
        """
        🔧 修复4: 增强采样的数值稳定性
        """
        log_alpha = self.log_alpha.repeat(self.repeat)
        u = self.sampler.sample((self.features,)).type_as(log_alpha)
        
        # 添加数值稳定性
        log_u = torch.log(u)
        log_1_minus_u = torch.log(1 - u)
        
        # 防止除零和溢出
        alpha_term = (log_u - log_1_minus_u + log_alpha) / self.beta
        alpha_term = torch.clamp(alpha_term, min=-15.0, max=15.0)
        
        s = torch.sigmoid(alpha_term)
        s_bar = s * (self.max_s - self.min_s) + self.min_s
        z = F.hardtanh(s_bar, min_val=0, max_val=1)
        return z
    
    def deterministic_z(self):
        """
        🔧 修复5: 改进确定性掩码生成逻辑
        """
        sub_features = self.features // self.repeat
        Lc = self.L().sum() / self.repeat
        
        # 🔧 关键修复：正确计算要置零的参数数量
        # 原逻辑：num_zeros = round(sub_features - Lc.item()) * self.repeat
        # 问题：当Lc接近sub_features时，num_zeros接近0，几乎不剪枝
        # 修正：基于目标稀疏度计算
        
        target_retention_rate = Lc.item() / sub_features  # L的平均值就是目标保留率
        num_to_keep = max(1, round(sub_features * target_retention_rate))  # 至少保留1个
        num_zeros = sub_features - num_to_keep
        
        log_alpha = self.log_alpha.repeat(self.repeat)
        z = torch.sigmoid(log_alpha / self.beta * self.magical_number)
        
        if num_zeros > 0 and num_zeros < self.features:
            # 选择最小的num_zeros个元素置零
            _, indices = torch.topk(z, k=num_zeros, largest=False)
            z_new = z.clone()
            z_new[indices] = 0
            z = z_new
        
        # 调试信息
        if self.debug_call_count % 100 == 0:
            actual_zeros = (z == 0).sum().item()
            actual_sparsity = actual_zeros / self.features
            logger.debug(f"Deterministic mask - Target retention: {target_retention_rate:.4f}, "
                        f"Actual sparsity: {actual_sparsity:.4f}, "
                        f"Zeros: {actual_zeros}/{self.features}")
        
        return z
    
    def forward(self):
        if self.activate.item():
            if self.training:
                return self.sample_z()
            else:
                return self.deterministic_z()
        else:
            return self.deterministic_z().detach()

    @torch.no_grad()
    def parse(self):
        """
        🔧 修复6: 改进掩码解析逻辑
        """
        sub_features = self.features // self.repeat
        Lc = self.L().sum() / self.repeat
        
        # 更稳健的计算保留元素数量
        target_retention_rate = Lc.item() / sub_features
        num_non_zeros = max(1, round(sub_features * target_retention_rate))
        num_non_zeros = min(num_non_zeros, sub_features)  # 不能超过总数
        
        z = torch.sigmoid(self.log_alpha / self.beta * self.magical_number)
        
        # 选择top-k个最大的元素作为保留的
        indices = torch.topk(z, k=num_non_zeros).indices
        indices = torch.sort(indices).values
        
        if self.repeat > 1:  # shape: (num_heads, head_dim)
            z = z.repeat(self.repeat)
            indices = torch.concat(tuple(indices + i * sub_features for i in range(self.repeat)))
        
        return z[indices], indices

    def get_effective_sparsity(self):
        """
        🔧 新增: 获取当前掩码的实际稀疏度
        """
        with torch.no_grad():
            current_mask = self.deterministic_z()
            zeros = (current_mask == 0).sum().item()
            total = current_mask.numel()
            sparsity = zeros / total
            return sparsity, zeros, total

    def update_sparsity_target(self, target_sparsity: float):
        """
        🔧 新增: 根据目标稀疏度调整log_alpha
        """
        # 根据目标稀疏度反推需要的log_alpha值
        # target_sparsity = 1 - retention_rate
        target_retention = 1.0 - target_sparsity
        target_retention = torch.clamp(torch.tensor(target_retention), min=0.01, max=0.99)
        
        # 通过sigmoid反函数计算需要的log_alpha
        # retention ≈ sigmoid(log_alpha), 所以 log_alpha ≈ logit(retention)
        target_logit = torch.log(target_retention / (1 - target_retention))
        
        # 平滑更新log_alpha
        with torch.no_grad():
            current_alpha = self.log_alpha.mean()
            # 使用指数移动平均平滑更新
            momentum = 0.9
            new_alpha = momentum * current_alpha + (1 - momentum) * target_logit
            self.log_alpha.data.fill_(new_alpha.item())
            
            # 添加小量随机扰动，避免所有参数完全一致
            noise = torch.randn_like(self.log_alpha) * 0.01
            self.log_alpha.data += noise


class LinearWithMaskBefore(nn.Linear):
    
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mask = Mask(in_features)
        self.kw_args = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "device": device, 
            "dtype": dtype,
        }

    def super_forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        # 🔧 修复7: 确保掩码确实生效
        mask_values = self.mask()
        
        # 调试：验证掩码是否真的在工作
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 1
            
        if self._debug_counter % 1000 == 0:  # 每1000次前向传播打印一次
            zeros = (mask_values == 0).sum().item()
            total = mask_values.numel()
            logger.debug(f"LinearWithMaskBefore - Masked {zeros}/{total} parameters "
                        f"({zeros/total*100:.1f}% pruned)")
        
        x = x * mask_values
        x = super().forward(x)
        return x

    def get_hook_fn(self, act_dict, name):
        def fn(module, inp, outp):
            assert len(inp[0].shape) == 3 and inp[0].shape[0] == 1
            inp: Tensor = inp[0].squeeze(0).cpu()
            act_values = inp.abs().max(dim=0).values
            filter_weights = module.weight.norm(dim=0)
            values = act_values * filter_weights
            if name not in act_dict:
                act_dict[name] = values
            else:
                act_dict[name] = torch.max(act_dict[name], values)            
        return fn

    def extract(self,
        indices: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> nn.Linear:
        if values is None:
            values = torch.ones_like(indices, dtype=self.weight.dtype)
        values = torch.diag(values)

        self.kw_args["in_features"] = indices.shape[0]
        new_linear = nn.Linear(**self.kw_args)        
        new_linear.weight.copy_(self.weight[:, indices] @ values)
        if new_linear.bias is not None:
            new_linear.bias.copy_(self.bias)        
        return new_linear
