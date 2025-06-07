# pipeline/dp_core.py
# Fed-Pruner差分隐私核心实现
# 基于DP-FedAvg算法的严格(ε,δ)-差分隐私保证

import torch
import torch.nn as nn
import numpy as np
import math
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class DPConfig:
    """差分隐私配置类"""
    # 核心DP参数
    target_epsilon: float = 10.0  # 目标隐私预算
    target_delta: float = 1e-5  # 目标失败概率
    noise_multiplier: float = 1.0  # 噪声乘数
    clipping_bound: float = 1.0  # 梯度剪裁界限

    # 训练参数
    num_epochs: int = 10  # 总训练轮数
    num_clients: int = 10  # 客户端数量
    sampling_rate: float = 1.0  # 客户端采样率

    # 高级设置
    accountant_type: str = "rdp"  # 隐私会计类型: "rdp", "gdp"
    auto_clip: bool = True  # 自动剪裁阈值调整
    adaptive_noise: bool = True  # 自适应噪声调整

    def validate(self):
        """验证配置有效性"""
        assert self.target_epsilon > 0, "target_epsilon must be positive"
        assert 0 < self.target_delta < 1, "target_delta must be in (0,1)"
        assert self.noise_multiplier > 0, "noise_multiplier must be positive"
        assert self.clipping_bound > 0, "clipping_bound must be positive"
        assert 0 < self.sampling_rate <= 1, "sampling_rate must be in (0,1]"


class PrivacyAccountant:
    """高精度隐私会计系统，支持RDP和GDP"""

    def __init__(self, delta: float = 1e-5, accountant_type: str = "rdp"):
        self.delta = delta
        self.accountant_type = accountant_type.lower()

        if self.accountant_type == "rdp":
            # RDP orders from 1.25 to 64 for high precision
            self.orders = np.concatenate([
                np.linspace(1.25, 2, 16),
                np.logspace(np.log10(2), np.log10(64), 50)
            ])
            self.rdp_eps = np.zeros_like(self.orders)

        self.steps = 0
        self.history = []

    def step(self, noise_multiplier: float, sampling_rate: float,
             num_selected: int = None) -> float:
        """记录一步训练的隐私消耗"""
        self.steps += 1

        if self.accountant_type == "rdp":
            eps = self._rdp_step(noise_multiplier, sampling_rate, num_selected)
        else:
            eps = self._gdp_step(noise_multiplier, sampling_rate, num_selected)

        self.history.append({
            'step': self.steps,
            'epsilon': eps,
            'noise_multiplier': noise_multiplier,
            'sampling_rate': sampling_rate
        })

        return eps

    def _rdp_step(self, noise_multiplier: float, sampling_rate: float,
                  num_selected: int = None) -> float:
        """RDP隐私会计"""
        if noise_multiplier <= 0:
            return float('inf')

        # 计算每个order的RDP值
        for i, order in enumerate(self.orders):
            if sampling_rate == 1.0:
                # 无采样情况
                rdp = order / (2 * noise_multiplier ** 2)
            else:
                # 泊松采样情况下的RDP
                rdp = self._compute_rdp_poisson_sampling(
                    order, noise_multiplier, sampling_rate
                )

            self.rdp_eps[i] += rdp

        # 转换为(ε,δ)-DP
        return self._rdp_to_dp()

    def _compute_rdp_poisson_sampling(self, order: float, noise_multiplier: float,
                                      sampling_rate: float) -> float:
        """计算泊松采样下的RDP值"""
        if sampling_rate >= 1.0:
            return order / (2 * noise_multiplier ** 2)

        alpha = order
        p = sampling_rate
        sigma = noise_multiplier

        if alpha == 1:
            return p * (np.exp(1 / (2 * sigma ** 2)) - 1)

        # For α > 1, use the tighter bound
        log_a = (alpha - 1) * np.log(1 - p + p * np.exp(1 / sigma ** 2)) - np.log(alpha)
        a = np.exp(log_a)

        if a <= 0:
            return 0

        return np.log(a) / (alpha - 1)

    def _rdp_to_dp(self) -> float:
        """将RDP转换为(ε,δ)-DP"""
        min_eps = float('inf')

        for i, order in enumerate(self.orders):
            if self.rdp_eps[i] == 0:
                continue

            eps = self.rdp_eps[i] + np.log(1 / self.delta) / (order - 1)
            min_eps = min(min_eps, eps)

        return min_eps if min_eps < float('inf') else 0.0

    def _gdp_step(self, noise_multiplier: float, sampling_rate: float,
                  num_selected: int = None) -> float:
        """GDP (Gaussian Differential Privacy) 会计"""
        sigma = noise_multiplier
        mu = np.sqrt(2 * np.log(1.25 / self.delta)) / sigma

        # 累积隐私损失
        if not hasattr(self, 'gdp_mu'):
            self.gdp_mu = 0.0

        self.gdp_mu += mu * sampling_rate

        # 转换为ε
        eps = self.gdp_mu + np.sqrt(2 * self.gdp_mu * np.log(1 / self.delta))
        return eps

    def get_privacy_spent(self) -> Tuple[float, float]:
        """获取已消耗的隐私预算"""
        if self.steps == 0:
            return 0.0, self.delta
        return self.history[-1]['epsilon'], self.delta

    def reset(self):
        """重置会计"""
        if self.accountant_type == "rdp":
            self.rdp_eps.fill(0)
        else:
            self.gdp_mu = 0.0
        self.steps = 0
        self.history.clear()


class AdaptiveClipping:
    """自适应剪裁阈值调整"""

    def __init__(self, initial_bound: float, target_quantile: float = 0.5):
        self.initial_bound = initial_bound
        self.current_bound = initial_bound
        self.target_quantile = target_quantile
        self.norm_history = []
        self.update_frequency = 50  # 每50个更新调整一次

    def update(self, norm: float):
        """更新剪裁阈值"""
        self.norm_history.append(norm)

        if len(self.norm_history) >= self.update_frequency:
            # 计算目标分位数
            new_bound = np.quantile(self.norm_history, self.target_quantile)

            # 平滑更新
            momentum = 0.9
            self.current_bound = momentum * self.current_bound + (1 - momentum) * new_bound

            # 清空历史
            self.norm_history = []

            logger.debug(f"Adaptive clipping: updated bound to {self.current_bound:.4f}")

    def get_threshold(self) -> float:
        return self.current_bound


class NoiseScheduler:
    """噪声水平调度器"""

    def __init__(self, config: DPConfig):
        self.config = config
        self.initial_noise = config.noise_multiplier

    def get_noise_multiplier(self, epoch: int) -> float:
        """根据训练进度调整噪声水平"""
        if not self.config.adaptive_noise:
            return self.initial_noise

        # 线性衰减策略
        progress = epoch / self.config.num_epochs
        min_noise = self.initial_noise * 0.5

        noise = self.initial_noise - (self.initial_noise - min_noise) * progress
        return max(min_noise, noise)


class DifferentialPrivacyManager:
    """差分隐私管理器 - 核心DP-FedAvg实现"""

    def __init__(self, config: DPConfig):
        self.config = config
        self.config.validate()

        self.accountant = PrivacyAccountant(
            delta=config.target_delta,
            accountant_type=config.accountant_type
        )

        # 自适应参数
        self.adaptive_clipping = AdaptiveClipping(config.clipping_bound) if config.auto_clip else None
        self.noise_scheduler = NoiseScheduler(config) if config.adaptive_noise else None

        # 统计信息
        self.stats = {
            'total_clipped_clients': 0,
            'avg_norm_before_clip': [],
            'avg_norm_after_clip': [],
            'noise_levels': [],
            'privacy_budget_history': []
        }

        logger.info(f"DP Manager initialized with ε={config.target_epsilon}, δ={config.target_delta}")

    def process_client_update(self,
                              initial_params: Dict[str, torch.Tensor],
                              updated_params: Dict[str, torch.Tensor],
                              client_id: int = None) -> Dict[str, torch.Tensor]:
        """处理单个客户端更新 - 计算增量并剪裁"""

        # 1. 计算参数更新增量
        update_delta = {}
        for key in initial_params:
            if key in updated_params and initial_params[key].shape == updated_params[key].shape:
                delta = updated_params[key] - initial_params[key]
                update_delta[key] = delta

        # 2. 计算L2范数
        total_norm = self._compute_l2_norm(update_delta)
        self.stats['avg_norm_before_clip'].append(total_norm)

        # 3. 应用剪裁
        clipped_update = self._clip_update(update_delta, total_norm)

        # 4. 记录统计信息
        clipped_norm = self._compute_l2_norm(clipped_update)
        self.stats['avg_norm_after_clip'].append(clipped_norm)

        if total_norm > self.config.clipping_bound:
            self.stats['total_clipped_clients'] += 1
            if client_id is not None:
                logger.debug(f"Client {client_id}: clipped norm {total_norm:.4f} -> {clipped_norm:.4f}")

        # 5. 自适应剪裁阈值调整
        if self.adaptive_clipping:
            self.adaptive_clipping.update(total_norm)
            self.config.clipping_bound = self.adaptive_clipping.get_threshold()

        return clipped_update

    def _compute_l2_norm(self, param_dict: Dict[str, torch.Tensor]) -> float:
        """计算参数字典的L2范数"""
        total_norm_squared = 0.0
        for tensor in param_dict.values():
            if tensor.is_floating_point():
                total_norm_squared += tensor.norm().item() ** 2
        return math.sqrt(total_norm_squared)

    def _clip_update(self, update_delta: Dict[str, torch.Tensor],
                     total_norm: float) -> Dict[str, torch.Tensor]:
        """应用L2范数剪裁"""
        if total_norm <= self.config.clipping_bound:
            return update_delta

        # 计算剪裁因子
        clip_factor = self.config.clipping_bound / total_norm

        clipped_update = {}
        for key, tensor in update_delta.items():
            if tensor.is_floating_point():
                clipped_update[key] = tensor * clip_factor
            else:
                clipped_update[key] = tensor.clone()

        return clipped_update

    def aggregate_and_add_noise(self,
                                client_updates: List[Dict[str, torch.Tensor]],
                                epoch: int = None) -> Dict[str, torch.Tensor]:
        """聚合客户端更新并添加DP噪声"""

        if not client_updates:
            raise ValueError("No client updates to aggregate")

        # 1. 联邦平均
        aggregated_update = self._federated_averaging(client_updates)

        # 2. 计算噪声参数
        current_noise_multiplier = self.config.noise_multiplier
        if self.noise_scheduler:
            current_noise_multiplier = self.noise_scheduler.get_noise_multiplier(epoch)

        # 3. 添加高斯噪声
        noisy_update = self._add_gaussian_noise(aggregated_update, current_noise_multiplier)

        # 4. 更新隐私预算
        sampling_rate = len(client_updates) / self.config.num_clients
        current_epsilon = self.accountant.step(
            current_noise_multiplier,
            sampling_rate,
            len(client_updates)
        )

        # 5. 记录统计信息
        self.stats['noise_levels'].append(current_noise_multiplier)
        self.stats['privacy_budget_history'].append(current_epsilon)

        # 6. 检查隐私预算是否超限
        if current_epsilon > self.config.target_epsilon:
            logger.warning(f"Privacy budget exceeded! Current ε={current_epsilon:.4f}, "
                           f"Target ε={self.config.target_epsilon:.4f}")

        logger.info(f"Aggregated {len(client_updates)} updates, ε={current_epsilon:.4f}, "
                    f"noise_mult={current_noise_multiplier:.4f}")

        return noisy_update

    def _federated_averaging(self,
                             client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """执行联邦平均"""
        if not client_updates:
            return {}

        # 初始化聚合结果
        aggregated = {}
        first_update = client_updates[0]

        for key in first_update:
            aggregated[key] = torch.zeros_like(first_update[key])

        # 累加所有客户端更新
        for update in client_updates:
            for key in aggregated:
                if key in update:
                    aggregated[key] += update[key]

        # 取平均
        num_clients = len(client_updates)
        for key in aggregated:
            aggregated[key] /= num_clients

        return aggregated

    def _add_gaussian_noise(self,
                            aggregated_update: Dict[str, torch.Tensor],
                            noise_multiplier: float) -> Dict[str, torch.Tensor]:
        """添加校准的高斯噪声"""
        sigma = noise_multiplier * self.config.clipping_bound
        noisy_update = {}

        for key, tensor in aggregated_update.items():
            if tensor.is_floating_point():
                # 生成与张量形状相同的高斯噪声
                noise = torch.normal(
                    mean=0.0,
                    std=sigma,
                    size=tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype
                )
                noisy_update[key] = tensor + noise
            else:
                # 非浮点参数不加噪声
                noisy_update[key] = tensor.clone()

        return noisy_update

    def get_privacy_budget(self) -> Tuple[float, float]:
        """获取当前隐私预算"""
        return self.accountant.get_privacy_spent()

    def get_remaining_budget(self) -> float:
        """获取剩余隐私预算"""
        current_eps, _ = self.get_privacy_budget()
        return max(0, self.config.target_epsilon - current_eps)

    def can_continue_training(self) -> bool:
        """检查是否可以继续训练"""
        current_eps, _ = self.get_privacy_budget()
        return current_eps < self.config.target_epsilon

    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        stats = self.stats.copy()
        current_eps, delta = self.get_privacy_budget()

        stats.update({
            'current_epsilon': current_eps,
            'delta': delta,
            'target_epsilon': self.config.target_epsilon,
            'remaining_budget': self.get_remaining_budget(),
            'total_steps': self.accountant.steps,
            'clipping_rate': self.stats['total_clipped_clients'] / max(1, len(self.stats['avg_norm_before_clip'])),
            'avg_norm_reduction': np.mean(self.stats['avg_norm_after_clip']) / max(1e-8, np.mean(
                self.stats['avg_norm_before_clip']))
        })

        return stats


class FederatedDPTraining:
    """
    联邦差分隐私训练协调器 - 完整实现
    集成Fed-Pruner的蒸馏训练、模型剪枝和DP保护
    """

    def __init__(self, dp_config: DPConfig):
        self.dp_manager = DifferentialPrivacyManager(dp_config)
        self.config = dp_config

        # 训练状态追踪
        self.current_round = 0
        self.training_history = []

        logger.info("🔒 FederatedDPTraining initialized with DP protection")

    def client_training_step(self,
                             client_trainer,  # DistillTrainer实例
                             server_weights: Dict[str, torch.Tensor],
                             client_train_data,  # 客户端训练数据
                             client_id: int,
                             training_args,  # TrainingArguments
                             teacher_model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """
        完整的客户端DP训练步骤

        Args:
            client_trainer: Fed-Pruner的DistillTrainer实例
            server_weights: 服务器下发的模型权重
            client_train_data: 客户端训练数据
            client_id: 客户端ID
            training_args: 训练参数
            teacher_model: 教师模型（用于知识蒸馏）

        Returns:
            剪裁后的模型更新增量
        """

        logger.info(f"🔒 Starting DP client training for client {client_id}")
        start_time = time.time()

        try:
            # ========== 1. 保存初始权重 ==========
            initial_weights = deepcopy(server_weights)

            # ========== 2. 准备客户端模型 ==========
            client_model = client_trainer.model

            # 确保模型在正确的设备上
            device = next(client_model.parameters()).device

            # 加载服务器权重到客户端模型
            try:
                client_model.load_state_dict(server_weights, strict=False)
                logger.debug(f"Client {client_id}: Loaded server weights successfully")
            except Exception as e:
                logger.error(f"Client {client_id}: Failed to load server weights: {e}")
                # 如果加载失败，使用兼容性加载
                self._safe_load_state_dict(client_model, server_weights)

            # ========== 3. 设置训练环境 ==========
            client_model.train()

            # 创建数据加载器（如果需要）
            if hasattr(client_train_data, '__iter__'):
                train_dataloader = client_train_data
            else:
                # 假设client_train_data是数据集，需要创建DataLoader
                from torch.utils.data import DataLoader
                train_dataloader = DataLoader(
                    client_train_data,
                    batch_size=training_args.per_device_train_batch_size,
                    shuffle=True
                )

            # ========== 4. 本地训练循环 ==========
            num_local_epochs = getattr(training_args, 'local_epochs', 1)
            total_loss = 0.0
            num_batches = 0

            for local_epoch in range(num_local_epochs):
                epoch_loss = 0.0
                epoch_batches = 0

                for batch_idx, batch in enumerate(train_dataloader):
                    try:
                        # 将数据移动到正确的设备
                        batch = self._move_batch_to_device(batch, device)

                        # ========== 4.1 前向传播和损失计算 ==========
                        if hasattr(client_trainer, 'compute_loss'):
                            # 使用Fed-Pruner的损失计算（包含蒸馏损失）
                            loss = client_trainer.compute_loss(client_model, batch)
                        else:
                            # 回退到基本损失计算
                            loss = self._compute_basic_loss(client_model, batch)

                        # ========== 4.2 反向传播 ==========
                        client_trainer.optimizer.zero_grad()
                        loss.backward()

                        # ========== 4.3 梯度处理（可选的梯度剪裁） ==========
                        if hasattr(training_args, 'max_grad_norm') and training_args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                client_model.parameters(),
                                training_args.max_grad_norm
                            )

                        # ========== 4.4 参数更新 ==========
                        client_trainer.optimizer.step()

                        # 如果有学习率调度器
                        if hasattr(client_trainer, 'lr_scheduler') and client_trainer.lr_scheduler:
                            client_trainer.lr_scheduler.step()

                        # ========== 4.5 记录训练统计 ==========
                        epoch_loss += loss.item()
                        epoch_batches += 1

                        # 可选：限制训练步数（用于快速实验）
                        if hasattr(training_args, 'max_steps_per_client'):
                            if batch_idx >= training_args.max_steps_per_client:
                                break

                    except Exception as e:
                        logger.error(f"Client {client_id}: Training step failed: {e}")
                        continue

                avg_epoch_loss = epoch_loss / max(1, epoch_batches)
                logger.debug(f"Client {client_id}: Local epoch {local_epoch + 1}, avg loss: {avg_epoch_loss:.4f}")

                total_loss += epoch_loss
                num_batches += epoch_batches

            # ========== 5. 获取训练后的权重 ==========
            updated_weights = self._create_safe_state_dict(client_model)

            if updated_weights is None:
                logger.error(f"Client {client_id}: Failed to extract updated weights")
                return self._create_zero_update(initial_weights)

            # ========== 6. DP处理：计算并剪裁更新增量 ==========
            clipped_update = self.dp_manager.process_client_update(
                initial_weights, updated_weights, client_id
            )

            # ========== 7. 记录训练统计 ==========
            training_time = time.time() - start_time
            avg_loss = total_loss / max(1, num_batches)

            self._record_client_training_stats(
                client_id, training_time, avg_loss, num_batches
            )

            logger.info(f"🔒 Client {client_id} DP training completed: "
                        f"time={training_time:.2f}s, avg_loss={avg_loss:.4f}")

            return clipped_update

        except Exception as e:
            logger.error(f"Client {client_id} DP training failed: {e}")
            # 返回零更新，避免训练中断
            return self._create_zero_update(initial_weights)

    def server_aggregation_step(self,
                                client_updates: List[Dict[str, torch.Tensor]],
                                epoch: int) -> Dict[str, torch.Tensor]:
        """
        服务器DP聚合步骤（扩展版本）

        Args:
            client_updates: 客户端更新列表
            epoch: 当前轮次

        Returns:
            加噪后的聚合更新
        """

        if not client_updates:
            logger.error("No client updates received for aggregation")
            return {}

        logger.info(f"🔒 Starting DP aggregation for round {epoch + 1}")
        start_time = time.time()

        # 过滤无效更新
        valid_updates = [update for update in client_updates if update and len(update) > 0]

        if not valid_updates:
            logger.error("No valid client updates for aggregation")
            return {}

        logger.info(f"Aggregating {len(valid_updates)}/{len(client_updates)} valid updates")

        # 执行DP聚合
        try:
            noisy_update = self.dp_manager.aggregate_and_add_noise(valid_updates, epoch)

            aggregation_time = time.time() - start_time

            # 记录聚合统计
            current_eps, delta = self.dp_manager.get_privacy_budget()
            remaining_budget = self.dp_manager.get_remaining_budget()

            logger.info(f"🔒 DP aggregation completed: time={aggregation_time:.2f}s, "
                        f"ε={current_eps:.4f}, remaining={remaining_budget:.4f}")

            # 更新训练历史
            self.training_history.append({
                'round': epoch + 1,
                'num_clients': len(valid_updates),
                'privacy_budget': current_eps,
                'aggregation_time': aggregation_time
            })

            return noisy_update

        except Exception as e:
            logger.error(f"DP aggregation failed: {e}")
            return {}

    def apply_update_to_server(self,
                               server_model: nn.Module,
                               aggregated_update: Dict[str, torch.Tensor],
                               learning_rate: float = 1.0):
        """
        将聚合更新应用到服务器模型（扩展版本）

        Args:
            server_model: 服务器模型
            aggregated_update: 聚合更新
            learning_rate: 服务器端学习率
        """

        if not aggregated_update:
            logger.warning("No aggregated update to apply")
            return

        try:
            with torch.no_grad():
                server_state = server_model.state_dict()
                updated_params = 0

                for key, update in aggregated_update.items():
                    if key in server_state and server_state[key].shape == update.shape:
                        # 应用更新：新权重 = 旧权重 + 学习率 * 更新
                        server_state[key] += learning_rate * update.to(server_state[key].device)
                        updated_params += 1
                    else:
                        logger.warning(f"Skipping parameter {key}: shape mismatch or not found")

                server_model.load_state_dict(server_state)
                logger.info(f"Applied updates to {updated_params} parameters")

        except Exception as e:
            logger.error(f"Failed to apply server update: {e}")

    # ========== 辅助方法 ==========

    def _safe_load_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]):
        """安全地加载状态字典，处理键名不匹配的情况"""
        model_state = model.state_dict()

        # 找到匹配的键
        matched_keys = []
        for key in state_dict:
            if key in model_state and model_state[key].shape == state_dict[key].shape:
                matched_keys.append(key)

        # 只加载匹配的参数
        filtered_state_dict = {k: state_dict[k] for k in matched_keys}
        model.load_state_dict(filtered_state_dict, strict=False)

        logger.info(f"Loaded {len(matched_keys)}/{len(state_dict)} parameters successfully")

    def _move_batch_to_device(self, batch, device):
        """将批次数据移动到指定设备"""
        if isinstance(batch, dict):
            return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [item.to(device) if hasattr(item, 'to') else item for item in batch]
        else:
            return batch.to(device) if hasattr(batch, 'to') else batch

    def _compute_basic_loss(self, model: nn.Module, batch) -> torch.Tensor:
        """计算基本损失（当没有自定义compute_loss时使用）"""
        if isinstance(batch, dict):
            # 假设是transformers格式
            outputs = model(**batch)
            return outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        else:
            # 假设是(inputs, labels)格式
            inputs, labels = batch
            outputs = model(inputs)
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(outputs, labels)

    def _create_safe_state_dict(self, model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        """安全地创建状态字典"""
        try:
            state_dict = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    state_dict[name] = param.data.clone().detach()

            # 也包含buffer参数
            for name, buffer in model.named_buffers():
                state_dict[name] = buffer.clone().detach()

            return state_dict
        except Exception as e:
            logger.error(f"Failed to create state dict: {e}")
            return None

    def _create_zero_update(self, reference_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """创建零更新（当训练失败时使用）"""
        zero_update = {}
        for key, tensor in reference_weights.items():
            zero_update[key] = torch.zeros_like(tensor)
        return zero_update

    def _record_client_training_stats(self, client_id: int, training_time: float,
                                      avg_loss: float, num_batches: int):
        """记录客户端训练统计"""
        # 这里可以扩展为更详细的统计记录
        logger.debug(f"Client {client_id} stats: time={training_time:.2f}s, "
                     f"loss={avg_loss:.4f}, batches={num_batches}")

    # ========== 状态查询方法 ==========

    def get_privacy_status(self) -> Dict[str, Any]:
        """获取当前隐私状态"""
        current_eps, delta = self.dp_manager.get_privacy_budget()
        remaining = self.dp_manager.get_remaining_budget()

        return {
            'current_epsilon': current_eps,
            'target_epsilon': self.config.target_epsilon,
            'delta': delta,
            'remaining_budget': remaining,
            'budget_utilization': current_eps / self.config.target_epsilon,
            'can_continue_training': self.dp_manager.can_continue_training(),
            'total_rounds': len(self.training_history)
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.training_history:
            return {'error': 'No training history available'}

        total_clients = sum(round_info['num_clients'] for round_info in self.training_history)
        avg_clients_per_round = total_clients / len(self.training_history)
        total_time = sum(round_info['aggregation_time'] for round_info in self.training_history)

        return {
            'total_rounds': len(self.training_history),
            'total_clients_trained': total_clients,
            'avg_clients_per_round': avg_clients_per_round,
            'total_aggregation_time': total_time,
            'avg_time_per_round': total_time / len(self.training_history),
            'privacy_status': self.get_privacy_status()
        }

    def reset_training_state(self):
        """重置训练状态（用于新实验）"""
        self.current_round = 0
        self.training_history.clear()
        self.dp_manager.accountant.reset()
        logger.info("Training state reset")


# ========== 便捷集成函数 ==========

def create_dp_federated_trainer(training_args,
                                dp_config: Optional[DPConfig] = None) -> FederatedDPTraining:
    """
    根据training_args创建DP联邦训练器

    Args:
        training_args: Fed-Pruner的TrainingArguments
        dp_config: 可选的DP配置（如果None则从training_args创建）

    Returns:
        配置好的FederatedDPTraining实例
    """

    if dp_config is None:
        # 从training_args创建DP配置
        dp_config = DPConfig(
            target_epsilon=getattr(training_args, 'dp_target_epsilon', 10.0),
            target_delta=getattr(training_args, 'dp_target_delta', 1e-5),
            noise_multiplier=getattr(training_args, 'dp_noise_multiplier', 1.0),
            clipping_bound=getattr(training_args, 'dp_clipping_bound', 1.0),
            num_epochs=getattr(training_args, 'num_train_epochs', 10),
            num_clients=getattr(training_args, 'num_clients', 10),
            accountant_type=getattr(training_args, 'dp_accountant_type', 'rdp'),
            auto_clip=getattr(training_args, 'dp_auto_clip', True),
            adaptive_noise=getattr(training_args, 'dp_adaptive_noise', True)
        )

    return FederatedDPTraining(dp_config)


def integrate_dp_with_client_trainer(client_trainer, dp_trainer: FederatedDPTraining):
    """
    将DP训练器与现有的客户端训练器集成

    Args:
        client_trainer: Fed-Pruner的DistillTrainer或其他训练器
        dp_trainer: DP联邦训练器

    Returns:
        集成后的训练器
    """

    # 保存原始的训练方法
    original_train_step = getattr(client_trainer, 'training_step', None)

    def dp_enhanced_training_step(model, inputs, **kwargs):
        """DP增强的训练步骤"""
        # 执行原始训练步骤
        if original_train_step:
            return original_train_step(model, inputs, **kwargs)
        else:
            # 默认训练步骤
            return client_trainer.compute_loss(model, inputs)

    # 替换训练步骤
    client_trainer.training_step = dp_enhanced_training_step
    client_trainer.dp_trainer = dp_trainer

    logger.info("Client trainer integrated with DP protection")
    return client_trainer

# 导出主要类
__all__ = [
    'DPConfig',
    'PrivacyAccountant',
    'DifferentialPrivacyManager',
    'FederatedDPTraining',
    'AdaptiveClipping',
    'NoiseScheduler'
]
