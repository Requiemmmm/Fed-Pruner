# pipeline/dp_utils.py
# Fed-Pruner差分隐私工具函数和辅助组件
# 包含验证、分析、可视化和调试工具

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, OrderedDict
from copy import deepcopy
import warnings
from dataclasses import asdict

from .dp_core import DPConfig, DifferentialPrivacyManager, PrivacyAccountant

logger = logging.getLogger(__name__)


# ================================ 验证工具 ================================

def validate_dp_implementation(dp_manager: DifferentialPrivacyManager,
                               num_test_rounds: int = 5) -> Dict[str, bool]:
    """
    验证DP实现的正确性

    Args:
        dp_manager: DP管理器实例
        num_test_rounds: 测试轮数

    Returns:
        验证结果字典
    """
    results = {}

    logger.info("🧪 Validating DP implementation...")

    # 测试1: 剪裁机制
    logger.info("Testing clipping mechanism...")
    try:
        initial_params = {"test_param": torch.randn(10, 10) * 0.1}
        large_update_params = {"test_param": torch.randn(10, 10) * 10.0}

        clipped_update = dp_manager.process_client_update(
            initial_params, large_update_params, 0
        )

        norm = np.sqrt(sum(tensor.norm().item() ** 2 for tensor in clipped_update.values()))
        results['clipping_works'] = norm <= dp_manager.config.clipping_bound + 1e-6

        if results['clipping_works']:
            logger.info(f"✅ Clipping works: norm={norm:.4f} ≤ bound={dp_manager.config.clipping_bound}")
        else:
            logger.error(f"❌ Clipping failed: norm={norm:.4f} > bound={dp_manager.config.clipping_bound}")

    except Exception as e:
        logger.error(f"❌ Clipping test failed: {e}")
        results['clipping_works'] = False

    # 测试2: 噪声添加
    logger.info("Testing noise addition...")
    try:
        client_updates = []
        for i in range(3):
            update = {"param": torch.randn(5, 5) * 0.1}
            client_updates.append(update)

        noisy_update = dp_manager.aggregate_and_add_noise(client_updates, 0)

        results['noise_addition_works'] = (
                isinstance(noisy_update, dict) and
                all(isinstance(v, torch.Tensor) for v in noisy_update.values())
        )

        if results['noise_addition_works']:
            logger.info("✅ Noise addition works correctly")
        else:
            logger.error("❌ Noise addition failed")

    except Exception as e:
        logger.error(f"❌ Noise addition test failed: {e}")
        results['noise_addition_works'] = False

    # 测试3: 隐私预算追踪
    logger.info("Testing privacy budget tracking...")
    try:
        initial_eps, _ = dp_manager.get_privacy_budget()

        # 模拟几轮训练
        eps_values = [initial_eps]
        for round_num in range(num_test_rounds):
            eps = dp_manager.accountant.step(1.0, 1.0, dp_manager.config.num_clients)
            eps_values.append(eps)

        # 检查epsilon是否单调递增
        is_monotonic = all(eps_values[i] <= eps_values[i + 1] for i in range(len(eps_values) - 1))
        results['budget_tracking_works'] = is_monotonic and eps_values[-1] > initial_eps

        if results['budget_tracking_works']:
            logger.info(f"✅ Budget tracking works: ε increased from {initial_eps:.4f} to {eps_values[-1]:.4f}")
        else:
            logger.error("❌ Budget tracking failed")

    except Exception as e:
        logger.error(f"❌ Budget tracking test failed: {e}")
        results['budget_tracking_works'] = False

    # 综合结果
    all_passed = all(results.values())
    results['overall_passed'] = all_passed

    if all_passed:
        logger.info("🎉 All DP validation tests passed!")
    else:
        failed_tests = [test for test, passed in results.items() if not passed and test != 'overall_passed']
        logger.warning(f"⚠️  Failed tests: {failed_tests}")

    return results


def check_privacy_budget_consumption(dp_manager: DifferentialPrivacyManager) -> Dict[str, float]:
    """
    检查隐私预算消耗情况

    Returns:
        隐私预算统计信息
    """
    current_eps, delta = dp_manager.get_privacy_budget()
    remaining = dp_manager.get_remaining_budget()
    utilization = current_eps / dp_manager.config.target_epsilon

    budget_info = {
        'current_epsilon': current_eps,
        'target_epsilon': dp_manager.config.target_epsilon,
        'delta': delta,
        'remaining_budget': remaining,
        'utilization_percentage': utilization * 100,
        'can_continue': dp_manager.can_continue_training()
    }

    logger.info(f"Privacy budget status: ε={current_eps:.4f}/{dp_manager.config.target_epsilon:.4f} "
                f"({utilization:.1%} used)")

    return budget_info


# ================================ 分析工具 ================================

def analyze_clipping_statistics(dp_manager: DifferentialPrivacyManager) -> Dict[str, Any]:
    """
    分析剪裁统计信息

    Returns:
        剪裁分析结果
    """
    stats = dp_manager.get_statistics()

    if not stats['avg_norm_before_clip']:
        return {'error': 'No clipping data available'}

    norms_before = np.array(stats['avg_norm_before_clip'])
    norms_after = np.array(stats['avg_norm_after_clip'])

    analysis = {
        'total_updates': len(norms_before),
        'clipped_updates': stats['total_clipped_clients'],
        'clipping_rate': stats['clipping_rate'],
        'avg_norm_before': np.mean(norms_before),
        'avg_norm_after': np.mean(norms_after),
        'max_norm_before': np.max(norms_before),
        'min_norm_before': np.min(norms_before),
        'norm_reduction_ratio': stats['avg_norm_reduction'],
        'current_clipping_bound': dp_manager.config.clipping_bound
    }

    # 计算分位数
    analysis.update({
        'norm_percentiles': {
            'p25': np.percentile(norms_before, 25),
            'p50': np.percentile(norms_before, 50),
            'p75': np.percentile(norms_before, 75),
            'p90': np.percentile(norms_before, 90),
            'p95': np.percentile(norms_before, 95)
        }
    })

    return analysis


def compute_privacy_utility_curve(base_accuracy: float,
                                  epsilon_values: List[float],
                                  accuracy_drops: List[float]) -> Dict[str, List[float]]:
    """
    计算隐私-效用权衡曲线

    Args:
        base_accuracy: 基线准确率（无DP）
        epsilon_values: epsilon值列表
        accuracy_drops: 对应的准确率下降列表

    Returns:
        隐私-效用曲线数据
    """
    if len(epsilon_values) != len(accuracy_drops):
        raise ValueError("epsilon_values and accuracy_drops must have same length")

    dp_accuracies = [base_accuracy - drop for drop in accuracy_drops]
    privacy_levels = [1 / eps if eps > 0 else float('inf') for eps in epsilon_values]

    return {
        'epsilon_values': epsilon_values,
        'dp_accuracies': dp_accuracies,
        'accuracy_drops': accuracy_drops,
        'privacy_levels': privacy_levels,
        'base_accuracy': base_accuracy
    }


# ================================ 数据处理工具 ================================

def create_non_iid_data_split(dataset,
                              num_clients: int,
                              alpha: float = 0.5,
                              min_samples_per_client: int = 10) -> List[List[int]]:
    """
    使用Dirichlet分布创建Non-IID数据划分

    Args:
        dataset: 数据集
        num_clients: 客户端数量
        alpha: Dirichlet分布的浓度参数
        min_samples_per_client: 每个客户端最小样本数

    Returns:
        每个客户端的数据索引列表
    """
    # 获取标签
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        # 假设数据集有label字段
        labels = [item['label'] if isinstance(item, dict) else item[1] for item in dataset]

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # 按标签分组数据索引
    unique_labels = np.unique(labels)
    label_to_indices = {label: [] for label in unique_labels}

    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    # 对每个标签使用Dirichlet分布分配
    for label in unique_labels:
        indices = label_to_indices[label]
        np.random.shuffle(indices)

        # 生成Dirichlet分布的权重
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # 确保每个客户端至少有一些样本
        proportions = np.maximum(proportions, 1e-6)
        proportions = proportions / proportions.sum()

        # 根据权重分配样本
        start_idx = 0
        for client_id, prop in enumerate(proportions):
            end_idx = start_idx + int(prop * len(indices))
            if client_id == num_clients - 1:  # 最后一个客户端获得剩余所有样本
                end_idx = len(indices)

            client_indices[client_id].extend(indices[start_idx:end_idx])
            start_idx = end_idx

    # 确保每个客户端有最小样本数
    for client_id in range(num_clients):
        if len(client_indices[client_id]) < min_samples_per_client:
            logger.warning(f"Client {client_id} has only {len(client_indices[client_id])} samples, "
                           f"less than minimum {min_samples_per_client}")

    # 打乱每个客户端的数据
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    # 统计信息
    total_samples = sum(len(indices) for indices in client_indices)
    logger.info(f"Non-IID data split created:")
    logger.info(f"  - {num_clients} clients")
    logger.info(f"  - {total_samples} total samples")
    logger.info(f"  - Alpha: {alpha}")
    logger.info(f"  - Samples per client: {[len(indices) for indices in client_indices]}")

    return client_indices


def compute_data_heterogeneity_metrics(client_data_indices: List[List[int]],
                                       labels: Union[List, np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    计算数据异构性指标

    Args:
        client_data_indices: 每个客户端的数据索引
        labels: 数据标签

    Returns:
        异构性指标
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    num_clients = len(client_data_indices)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 计算每个客户端的标签分布
    client_distributions = []
    for client_indices in client_data_indices:
        client_labels = labels[client_indices]
        label_counts = np.bincount(client_labels, minlength=num_classes)
        distribution = label_counts / len(client_indices) if len(client_indices) > 0 else np.zeros(num_classes)
        client_distributions.append(distribution)

    client_distributions = np.array(client_distributions)

    # 计算全局分布
    global_distribution = np.bincount(labels, minlength=num_classes) / len(labels)

    # KL散度（衡量与全局分布的差异）
    kl_divergences = []
    for i in range(num_clients):
        # 避免除零，添加小的平滑项
        client_dist = client_distributions[i] + 1e-10
        global_dist = global_distribution + 1e-10

        kl_div = np.sum(client_dist * np.log(client_dist / global_dist))
        kl_divergences.append(kl_div)

    # Earth Mover's Distance (Wasserstein距离)
    try:
        from scipy.stats import wasserstein_distance
        emd_distances = []
        for i in range(num_clients):
            emd = wasserstein_distance(
                range(num_classes), range(num_classes),
                client_distributions[i], global_distribution
            )
            emd_distances.append(emd)
        avg_emd = np.mean(emd_distances)
    except ImportError:
        logger.warning("scipy not available, skipping EMD calculation")
        avg_emd = None

    # Jensen-Shannon散度
    def js_divergence(p, q):
        p = p + 1e-10
        q = q + 1e-10
        m = 0.5 * (p + q)
        return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    js_divergences = []
    for i in range(num_clients):
        js_div = js_divergence(client_distributions[i], global_distribution)
        js_divergences.append(js_div)

    metrics = {
        'avg_kl_divergence': np.mean(kl_divergences),
        'max_kl_divergence': np.max(kl_divergences),
        'std_kl_divergence': np.std(kl_divergences),
        'avg_js_divergence': np.mean(js_divergences),
        'max_js_divergence': np.max(js_divergences),
        'num_clients': num_clients,
        'num_classes': num_classes,
        'samples_per_client': [len(indices) for indices in client_data_indices]
    }

    if avg_emd is not None:
        metrics['avg_emd'] = avg_emd

    return metrics


# ================================ 可视化工具 ================================

def plot_privacy_budget_over_time(privacy_history: List[float],
                                  target_epsilon: float,
                                  save_path: Optional[str] = None):
    """
    绘制隐私预算随时间变化的图表

    Args:
        privacy_history: 隐私预算历史
        target_epsilon: 目标epsilon值
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 6))

    epochs = list(range(len(privacy_history)))
    plt.plot(epochs, privacy_history, 'b-', linewidth=2, label='Privacy Budget (ε)')
    plt.axhline(y=target_epsilon, color='r', linestyle='--',
                label=f'Target ε = {target_epsilon}')

    plt.xlabel('Training Rounds')
    plt.ylabel('Privacy Budget (ε)')
    plt.title('Privacy Budget Consumption Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标注重要点
    if privacy_history:
        final_eps = privacy_history[-1]
        plt.annotate(f'Final ε = {final_eps:.3f}',
                     xy=(len(privacy_history) - 1, final_eps),
                     xytext=(len(privacy_history) - 1 - len(privacy_history) * 0.2, final_eps * 1.1),
                     arrowprops=dict(arrowstyle='->', color='black'))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Privacy budget plot saved to {save_path}")

    plt.show()


def plot_clipping_analysis(norms_before: List[float],
                           norms_after: List[float],
                           clipping_bound: float,
                           save_path: Optional[str] = None):
    """
    绘制剪裁分析图表

    Args:
        norms_before: 剪裁前的范数
        norms_after: 剪裁后的范数
        clipping_bound: 剪裁界限
        save_path: 保存路径（可选）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：范数分布
    ax1.hist(norms_before, bins=30, alpha=0.7, label='Before Clipping', color='blue')
    ax1.hist(norms_after, bins=30, alpha=0.7, label='After Clipping', color='orange')
    ax1.axvline(x=clipping_bound, color='red', linestyle='--',
                label=f'Clipping Bound = {clipping_bound}')
    ax1.set_xlabel('L2 Norm')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Update Norms')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：时间序列
    rounds = list(range(len(norms_before)))
    ax2.plot(rounds, norms_before, 'b-', alpha=0.7, label='Before Clipping')
    ax2.plot(rounds, norms_after, 'o-', alpha=0.7, label='After Clipping', markersize=3)
    ax2.axhline(y=clipping_bound, color='red', linestyle='--',
                label=f'Clipping Bound = {clipping_bound}')
    ax2.set_xlabel('Training Rounds')
    ax2.set_ylabel('L2 Norm')
    ax2.set_title('Update Norms Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Clipping analysis plot saved to {save_path}")

    plt.show()


def plot_data_distribution(client_data_indices: List[List[int]],
                           labels: Union[List, np.ndarray, torch.Tensor],
                           save_path: Optional[str] = None):
    """
    绘制客户端数据分布图

    Args:
        client_data_indices: 每个客户端的数据索引
        labels: 数据标签
        save_path: 保存路径（可选）
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    num_clients = len(client_data_indices)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 计算每个客户端的标签分布
    distribution_matrix = np.zeros((num_clients, num_classes))

    for i, client_indices in enumerate(client_data_indices):
        if len(client_indices) > 0:
            client_labels = labels[client_indices]
            label_counts = np.bincount(client_labels, minlength=num_classes)
            distribution_matrix[i] = label_counts / len(client_indices)

    # 创建热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(distribution_matrix,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=[f'Class {i}' for i in unique_labels],
                yticklabels=[f'Client {i}' for i in range(num_clients)])

    plt.title('Data Distribution Across Clients')
    plt.xlabel('Classes')
    plt.ylabel('Clients')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Data distribution plot saved to {save_path}")

    plt.show()


# ================================ 配置和日志工具 ================================

def save_dp_config(dp_config: DPConfig, save_path: str):
    """保存DP配置到文件"""
    config_dict = asdict(dp_config)

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"DP config saved to {save_path}")


def load_dp_config(config_path: str) -> DPConfig:
    """从文件加载DP配置"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return DPConfig(**config_dict)


def setup_dp_logging(log_level: str = "INFO",
                     log_file: Optional[str] = None) -> logging.Logger:
    """设置DP专用日志"""
    dp_logger = logging.getLogger('fed_pruner.dp')
    dp_logger.setLevel(getattr(logging, log_level.upper()))

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    dp_logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        dp_logger.addHandler(file_handler)

    return dp_logger


def create_dp_experiment_summary(dp_manager: DifferentialPrivacyManager,
                                 final_accuracy: float,
                                 baseline_accuracy: float,
                                 training_time: float) -> Dict[str, Any]:
    """创建DP实验总结"""
    stats = dp_manager.get_statistics()
    final_eps, delta = dp_manager.get_privacy_budget()

    summary = {
        'experiment_config': {
            'target_epsilon': dp_manager.config.target_epsilon,
            'target_delta': dp_manager.config.target_delta,
            'noise_multiplier': dp_manager.config.noise_multiplier,
            'clipping_bound': dp_manager.config.clipping_bound,
            'num_clients': dp_manager.config.num_clients,
            'num_epochs': dp_manager.config.num_epochs
        },
        'privacy_results': {
            'final_epsilon': final_eps,
            'delta': delta,
            'budget_utilization': final_eps / dp_manager.config.target_epsilon,
            'total_steps': stats['total_steps']
        },
        'performance_results': {
            'final_accuracy': final_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_drop': baseline_accuracy - final_accuracy,
            'relative_accuracy_drop': (baseline_accuracy - final_accuracy) / baseline_accuracy,
            'training_time_seconds': training_time
        },
        'clipping_statistics': {
            'total_clipped_clients': stats['total_clipped_clients'],
            'clipping_rate': stats['clipping_rate'],
            'avg_norm_reduction': stats['avg_norm_reduction']
        }
    }

    return summary


# ================================ 实验辅助工具 ================================

def generate_dp_experiment_configs(base_config: DPConfig,
                                   epsilon_values: List[float],
                                   noise_multipliers: List[float]) -> List[DPConfig]:
    """生成一系列DP实验配置"""
    configs = []

    for eps in epsilon_values:
        for noise_mult in noise_multipliers:
            config = deepcopy(base_config)
            config.target_epsilon = eps
            config.noise_multiplier = noise_mult
            configs.append(config)

    logger.info(f"Generated {len(configs)} experiment configurations")
    return configs


def estimate_training_time(dp_config: DPConfig,
                           time_per_round: float) -> Dict[str, float]:
    """估算训练时间"""
    total_rounds = dp_config.num_epochs
    estimated_time = total_rounds * time_per_round

    return {
        'total_rounds': total_rounds,
        'time_per_round_seconds': time_per_round,
        'estimated_total_time_seconds': estimated_time,
        'estimated_total_time_hours': estimated_time / 3600
    }


# 导出主要函数
__all__ = [
    'validate_dp_implementation',
    'check_privacy_budget_consumption',
    'analyze_clipping_statistics',
    'compute_privacy_utility_curve',
    'create_non_iid_data_split',
    'compute_data_heterogeneity_metrics',
    'plot_privacy_budget_over_time',
    'plot_clipping_analysis',
    'plot_data_distribution',
    'save_dp_config',
    'load_dp_config',
    'setup_dp_logging',
    'create_dp_experiment_summary',
    'generate_dp_experiment_configs',
    'estimate_training_time'
]
