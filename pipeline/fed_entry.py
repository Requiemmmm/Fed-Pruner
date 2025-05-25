import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import transformers
from transformers import (
    HfArgumentParser,
    DataCollatorWithPadding
)
from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification as TModel

from modeling.modeling_cofi_bert import CoFiBertForSequenceClassification as SModel

from datasets import DatasetDict, Dataset, load_from_disk, load_metric, load_dataset
from typing import Optional, Dict, List, Tuple, Callable, Union
from copy import deepcopy
from transformers import Trainer
import logging
from collections import OrderedDict

from .trainer import DistillTrainer
from .args import (
    TrainingArguments,
    ModelArguments,
)

# 量化工具导入（可选，防止导入错误）
try:
    from .quantization_utils import (
        quantize_model_dynamic,
        quantize_model_static,
        get_calibration_dataloader,
        simulate_quantization_communication_cost,
        dequantize_model_weights,
        get_model_size_info,
        QuantizationConfig
    )
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logging.warning("Quantization utilities not available. Quantization features disabled.")

from copy import deepcopy

Trainers = Union[Trainer, DistillTrainer]

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_hf_args():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    return args, training_args


def get_distill_args(args):
    distill_args = deepcopy(args)
    distill_args.num_train_epochs = args.distill_num_train_epochs
    distill_args.learning_rate = args.distill_learning_rate
    distill_args.evaluation_strategy = "epoch"

    return distill_args


def add_laplacian_noise_to_state_dict(state_dict, noise_scale_b):
    if not noise_scale_b > 0:
        return state_dict

    logger.info(f"Applying server-side Laplacian noise with scale b={noise_scale_b}")
    noisy_state_dict = OrderedDict()
    for key, param_tensor in state_dict.items():
        if param_tensor.is_floating_point():
            current_param_device = param_tensor.device
            current_param_dtype = param_tensor.dtype

            noise_values = np.random.laplace(loc=0.0, scale=noise_scale_b, size=param_tensor.shape)
            noise_tensor = torch.from_numpy(noise_values).to(device=current_param_device, dtype=current_param_dtype)

            noisy_state_dict[key] = param_tensor + noise_tensor
        else:
            noisy_state_dict[key] = param_tensor.clone()
    return noisy_state_dict


class Client():
    def __init__(self, epsilon=1000, num_clients=2):

        args, training_args = parse_hf_args()
        dataset = load_from_disk('./datasets/sst2')
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(example):
            return self.tokenizer(example["sentence"], truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)
        dataset['train'] = dataset['train'].filter(lambda x: len(x["input_ids"]) <= 512)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.epsilon = epsilon
        self.num_clients = num_clients
        self.dataset = dataset
        self.half = training_args.half
        self.client_train_datas = self.load_client_train_datas()

        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 1
        self.distill_args.gradient_accumulation_steps = 4
        
        # 量化配置初始化（保守设置）
        self.quantization_config = None
        if QUANTIZATION_AVAILABLE:
            try:
                if hasattr(training_args, 'apply_quantization') and training_args.apply_quantization:
                    self.quantization_config = QuantizationConfig(
                        apply_quantization=training_args.apply_quantization,
                        quantization_type=getattr(training_args, 'quantization_type', 'dynamic'),
                        quantization_backend=getattr(training_args, 'quantization_backend', 'fbgemm'),
                        calibration_batch_size=getattr(training_args, 'calibration_batch_size', 8),
                        num_calibration_batches=getattr(training_args, 'num_calibration_batches', 10)
                    )
                    # 保守的量化设置，避免影响训练稳定性
                    self.quantization_config.validate()
                    logger.info(f"Quantization enabled: {self.quantization_config.quantization_type}")
            except Exception as e:
                logger.warning(f"Quantization initialization failed: {e}")
                self.quantization_config = None

    def load_client_train_datas(self):
        client_train_datas = []
        if self.half == False:
            for i in range(self.num_clients):
                client_train_datas.append(
                    self.dataset['train'].shard(num_shards=self.num_clients, index=i, contiguous=True))
        else:
            for i in range(self.num_clients):
                client_train_datas.append(
                    self.dataset['train'].shard(num_shards=self.num_clients * 2, index=self.num_clients + i,
                                                contiguous=True))
        return client_train_datas

    def compute_metrics(self, eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return {"accuracy": accuracy}

    def _create_safe_state_dict(self, model):
        """创建安全的状态字典，确保所有参数都是叶子节点且数值稳定"""
        safe_state_dict = {}
        for key, param in model.state_dict().items():
            # 检查参数是否包含NaN或Inf
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"Parameter {key} contains NaN or Inf values, skipping client update")
                return None
                
            if param.requires_grad:
                # 创建新的叶子张量，并确保梯度信息正确
                safe_param = param.detach().clone()
                # 检查数值范围，防止梯度爆炸
                if safe_param.abs().max() > 100:
                    logger.warning(f"Parameter {key} has large values (max: {safe_param.abs().max()}), clipping")
                    safe_param = torch.clamp(safe_param, -10, 10)
                safe_param.requires_grad_(True)
                safe_state_dict[key] = safe_param
            else:
                safe_state_dict[key] = param.detach().clone()
        return safe_state_dict

    def _apply_quantization_simulation(self, server_model, client_id, datasets):
        """安全的量化模拟（仅用于分析，不影响训练）"""
        if not self.quantization_config or not self.quantization_config.apply_quantization:
            return
            
        try:
            logger.info(f"Client {client_id}: Running quantization simulation (analysis only)...")
            
            # 获取当前状态字典用于通信成本分析
            current_state_dict = server_model.state_dict()
            
            # 仅进行通信成本模拟，不实际量化训练中的模型
            if QUANTIZATION_AVAILABLE:
                comm_stats = simulate_quantization_communication_cost(current_state_dict)
                logger.info(f"Client {client_id}: Simulated communication reduction: {comm_stats['compression_ratio']:.2f}x")
                
                # 模型大小信息
                try:
                    model_info = get_model_size_info(server_model)
                    logger.info(f"Client {client_id}: Model size: {model_info['model_size_mb']:.2f} MB")
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Client {client_id}: Quantization simulation failed: {e}")

    def train_epoch(self, server_model, client_id, server_weights, t_model):
        datasets = self.client_train_datas[client_id]
        
        # 安全地加载权重
        try:
            server_model.load_state_dict(server_weights, strict=True)
        except Exception as e:
            logger.warning(f"Client {client_id}: Failed to load weights strictly: {e}")
            server_model.load_state_dict(server_weights, strict=False)

        # 检查模型参数的数值稳定性
        param_norms = []
        for name, param in server_model.named_parameters():
            if param.requires_grad:
                param_norm = param.norm().item()
                param_norms.append(param_norm)
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"Client {client_id}: Parameter {name} contains NaN/Inf before training")
                    return server_weights  # 返回原始权重，跳过这个客户端
                if param_norm > 50:  # 参数范数过大
                    logger.warning(f"Client {client_id}: Large parameter norm in {name}: {param_norm}")

        avg_param_norm = np.mean(param_norms) if param_norms else 0
        logger.info(f"Client {client_id}: Average parameter norm before training: {avg_param_norm:.4f}")

        # 确保模型参数是叶子节点
        for name, param in server_model.named_parameters():
            if param.requires_grad and not param.is_leaf:
                logger.warning(f"Client {client_id}: Fixing non-leaf parameter: {name}")
                param.data = param.data.detach().clone()
                param.requires_grad_(True)

        # 创建训练器时使用更保守的设置
        conservative_distill_args = deepcopy(self.distill_args)
        
        # 调整学习率，防止训练不稳定
        if hasattr(conservative_distill_args, 'learning_rate') and conservative_distill_args.learning_rate > 1e-4:
            conservative_distill_args.learning_rate = min(conservative_distill_args.learning_rate, 1e-4)
        if hasattr(conservative_distill_args, 'distill_learning_rate') and conservative_distill_args.distill_learning_rate > 1e-4:
            conservative_distill_args.distill_learning_rate = min(conservative_distill_args.distill_learning_rate, 1e-4)
            
        # 确保目标稀疏度不会过于激进
        if hasattr(conservative_distill_args, 'target_sparsity') and conservative_distill_args.target_sparsity > 0.9:
            conservative_distill_args.target_sparsity = 0.8
            logger.info(f"Client {client_id}: Adjusted target sparsity to {conservative_distill_args.target_sparsity}")

        distill_trainer = DistillTrainer(
            server_model,
            t_model,
            args=conservative_distill_args,
            train_dataset=datasets,
            eval_dataset=self.dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        try:
            # 确保训练前模型状态正确
            server_model.train()
            
            # 训练前检查一次模型输出
            with torch.no_grad():
                sample_input = datasets[0]
                inputs = self.tokenizer(sample_input['sentence'], return_tensors='pt', truncation=True, max_length=256)
                inputs = {k: v.to(next(server_model.parameters()).device) for k, v in inputs.items()}
                outputs = server_model(**inputs)
                if torch.isnan(outputs.logits).any():
                    logger.error(f"Client {client_id}: Model outputs NaN before training")
                    return server_weights

            # 执行训练
            distill_trainer.train()
            server_model.eval()
            
            # 训练后检查参数稳定性
            post_training_norms = []
            for name, param in server_model.named_parameters():
                if param.requires_grad:
                    param_norm = param.norm().item()
                    post_training_norms.append(param_norm)
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logger.error(f"Client {client_id}: Parameter {name} contains NaN/Inf after training")
                        return server_weights  # 返回原始权重
                    if param_norm > 100:  # 参数范数过大，可能梯度爆炸
                        logger.warning(f"Client {client_id}: Very large parameter norm in {name}: {param_norm}")

            avg_post_norm = np.mean(post_training_norms) if post_training_norms else 0
            logger.info(f"Client {client_id}: Average parameter norm after training: {avg_post_norm:.4f}")
            
            # 如果参数变化过大，可能训练不稳定
            if avg_post_norm > avg_param_norm * 10:
                logger.warning(f"Client {client_id}: Large parameter change detected, may be unstable")
            
            # 量化模拟（仅用于分析，不影响实际权重）
            self._apply_quantization_simulation(server_model, client_id, datasets)
            
            # 创建安全的权重字典
            weight = self._create_safe_state_dict(server_model)
            
            if weight is None:  # 如果检测到NaN/Inf
                logger.error(f"Client {client_id}: Unsafe weights detected, returning original weights")
                return server_weights
            
            logger.info(f"Client {client_id}: Training completed successfully")
            return weight
            
        except Exception as e:
            logger.error(f"Client {client_id}: Training failed: {e}")
            # 训练失败时返回原始权重，而不是破坏的权重
            logger.info(f"Client {client_id}: Returning original weights due to training failure")
            return server_weights


class Server():
    def __init__(self, epochs=10, num_clients=2):
        args, training_args = parse_hf_args()
        self.training_args = training_args
        self.num_clients = num_clients
        self.client = Client()
        self.epochs = epochs
        self.distill = training_args.distill

        if self.distill == True:
            self.t_model = TModel.from_pretrained('./[glue]/sst2-half-datas')
            self.s_model = SModel.from_pretrained('./[glue]/sst2-half-datas')
        if self.distill == False:
            self.t_model = TModel.from_pretrained('./model')
            self.s_model = SModel.from_pretrained('./model')

        dataset = load_from_disk('./datasets/sst2')
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(example):
            return self.tokenizer(example["sentence"], truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)
        dataset['validation'] = dataset['validation'].filter(lambda x: len(x["input_ids"]) <= 512)

        self.dataset = dataset['validation']

        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 1

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.best_result = 0
        
        # 添加准确率监控
        self.accuracy_history = []
        self.accuracy_drop_threshold = 0.1  # 如果准确率下降超过10%，触发警告
        
        # 量化状态记录
        self.quantization_enabled = (QUANTIZATION_AVAILABLE and 
                                   hasattr(training_args, 'apply_quantization') and 
                                   training_args.apply_quantization)
        if self.quantization_enabled:
            logger.info("Server: Quantization simulation enabled in federated learning")
            if QUANTIZATION_AVAILABLE:
                try:
                    initial_size_info = get_model_size_info(self.s_model)
                    logger.info(f"Server: Global model size: {initial_size_info['model_size_mb']:.2f}MB")
                except Exception as e:
                    logger.warning(f"Server: Could not get model size info: {e}")

    def distribute_task(self, client_ids):
        server_weights = deepcopy(self.s_model.state_dict())
        client_weight_datas = []

        for i in range(len(client_ids)):
            client_id = client_ids[i]
            weight = self.client.train_epoch(self.s_model, client_id, server_weights, self.t_model)
            client_weight_datas.append(weight)

        return client_weight_datas

    def _validate_client_weights(self, client_weight_datas):
        """验证客户端权重的有效性"""
        valid_weights = []
        for i, weights in enumerate(client_weight_datas):
            is_valid = True
            for key, param in weights.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.warning(f"Client {i} weights contain NaN/Inf in {key}, excluding from aggregation")
                    is_valid = False
                    break
            if is_valid:
                valid_weights.append(weights)
            else:
                logger.warning(f"Excluding client {i} from aggregation due to invalid weights")
        
        if len(valid_weights) == 0:
            logger.error("No valid client weights for aggregation! This will cause training failure.")
            return client_weight_datas  # 返回原始数据，让上层处理
        
        logger.info(f"Using {len(valid_weights)}/{len(client_weight_datas)} client weights for aggregation")
        return valid_weights

    def federated_average(self, client_weight_datas):
        # 验证客户端权重
        valid_client_weights = self._validate_client_weights(client_weight_datas)
        client_num = len(valid_client_weights)
        
        if client_num == 0:
            logger.error("No valid client weights for aggregation!")
            return self.s_model.state_dict()

        first_client_w = valid_client_weights[0]
        aggregated_w = OrderedDict()

        logger.info(f"Server: Starting federated aggregation with {client_num} clients...")
        
        for key in first_client_w.keys():
            param_template = first_client_w[key]

            if param_template.is_floating_point():
                # 聚合浮点张量（权重、偏置等）
                sum_param = torch.zeros_like(param_template)
                valid_param_count = 0
                
                for i in range(client_num):
                    client_param = valid_client_weights[i][key]
                    
                    # 再次检查参数有效性
                    if not (torch.isnan(client_param).any() or torch.isinf(client_param).any()):
                        sum_param += client_param.to(sum_param.device)
                        valid_param_count += 1

                if valid_param_count > 0:
                    aggregated_w[key] = sum_param / valid_param_count
                else:
                    # 如果所有客户端的这个参数都无效，保持服务器的原始参数
                    logger.warning(f"All client weights for {key} are invalid, keeping server weight")
                    aggregated_w[key] = self.s_model.state_dict()[key].clone()
            else:
                # 对于非浮点张量，使用第一个有效客户端的值
                aggregated_w[key] = param_template.clone()

        # 检查聚合后的权重
        for key, param in aggregated_w.items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"Aggregated weight {key} contains NaN/Inf! Using server's original weight.")
                aggregated_w[key] = self.s_model.state_dict()[key].clone()

        # 添加服务器端后处理噪声（减少噪声强度以提高稳定性）
        final_w = aggregated_w

        if hasattr(self.training_args, 'global_noise_type') and \
                self.training_args.global_noise_type.lower() == 'laplace' and \
                hasattr(self.training_args, 'global_noise_scale') and \
                self.training_args.global_noise_scale > 0:
            # 减少噪声强度以提高稳定性
            reduced_noise_scale = min(self.training_args.global_noise_scale, 0.01)
            final_w = add_laplacian_noise_to_state_dict(aggregated_w, reduced_noise_scale)
            logger.info(f"Applied reduced noise scale: {reduced_noise_scale}")

        # 加载最终权重到服务器模型
        try:
            self.s_model.load_state_dict(final_w, strict=True)
            logger.info("Server: Model weights updated successfully")
        except RuntimeError as e:
            logger.error(f"Failed to load state_dict with strict=True: {e}")
            try:
                self.s_model.load_state_dict(final_w, strict=False)
                logger.warning("Loaded state_dict with strict=False")
            except Exception as e2:
                logger.error(f"Failed to load state_dict even with strict=False: {e2}")
                logger.error("Model state may be corrupted!")

        logger.info("Server: Federated aggregation and model update complete.")
        return final_w

    def compute_metrics(self, eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return {"accuracy": accuracy}

    def _check_accuracy_drop(self, current_accuracy):
        """检查准确率是否异常下降"""
        self.accuracy_history.append(current_accuracy)
        
        if len(self.accuracy_history) >= 3:
            # 检查最近3轮的准确率趋势
            recent_accuracies = self.accuracy_history[-3:]
            if all(acc < 0.6 for acc in recent_accuracies):  # 连续3轮低于60%
                logger.warning("🚨 Accuracy consistently low! Possible training instability.")
                return True
                
            # 检查是否有急剧下降
            if len(self.accuracy_history) >= 2:
                prev_acc = self.accuracy_history[-2]
                if prev_acc - current_accuracy > self.accuracy_drop_threshold:
                    logger.warning(f"🚨 Significant accuracy drop: {prev_acc:.4f} -> {current_accuracy:.4f}")
                    return True
        
        return False

    def evalute(self):
        # 在评估前检查模型权重是否正常
        model_param_norms = []
        for name, param in self.s_model.named_parameters():
            if param.requires_grad:
                param_norm = param.norm().item()
                model_param_norms.append(param_norm)
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"Global model parameter {name} contains NaN/Inf!")
                    
        avg_norm = np.mean(model_param_norms) if model_param_norms else 0
        logger.info(f"Global model average parameter norm: {avg_norm:.4f}")

        distill_trainer = DistillTrainer(
            self.s_model,
            self.t_model,
            args=self.distill_args,
            eval_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        try:
            results = distill_trainer.evaluate(eval_dataset=self.dataset)
            current_accuracy = results['eval_accuracy']
            
            # 检查准确率异常下降
            accuracy_drop_detected = self._check_accuracy_drop(current_accuracy)
            if accuracy_drop_detected:
                logger.warning("Consider reducing learning rate or target sparsity")
            
            # 记录量化相关信息（如果启用）
            if self.quantization_enabled and QUANTIZATION_AVAILABLE:
                try:
                    current_size_info = get_model_size_info(self.s_model)
                    results['model_size_mb'] = current_size_info['model_size_mb']
                    logger.info(f"Global model size: {current_size_info['model_size_mb']:.2f}MB")
                except Exception as e:
                    logger.warning(f"Could not get current model size: {e}")
            
            if results['eval_accuracy'] > self.best_result and results['sparsity'] < 0.11:
                self.best_result = results['eval_accuracy']
            
            logger.info(f"Evaluation results: {results}")
            logger.info(f"Best results: {self.best_result}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # 如果评估失败，尝试简单的准确率计算
            try:
                self._simple_accuracy_check()
            except:
                logger.error("Even simple accuracy check failed!")

    def _simple_accuracy_check(self):
        """简单的准确率检查，用于评估失败时的备用方案"""
        logger.info("Running simple accuracy check...")
        self.s_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, sample in enumerate(self.dataset):
                if i >= 100:  # 只检查100个样本
                    break
                try:
                    inputs = self.tokenizer(sample['sentence'], return_tensors='pt', truncation=True, max_length=256)
                    inputs = {k: v.to(next(self.s_model.parameters()).device) for k, v in inputs.items()}
                    outputs = self.s_model(**inputs)
                    
                    if torch.isnan(outputs.logits).any():
                        logger.warning(f"Model output contains NaN for sample {i}")
                        continue
                        
                    prediction = torch.argmax(outputs.logits, dim=-1).item()
                    correct += (prediction == sample['label'])
                    total += 1
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
        
        simple_accuracy = correct / total if total > 0 else 0
        logger.info(f"Simple accuracy check: {simple_accuracy:.4f} ({correct}/{total})")

    def run(self):
        logger.info(f"Starting federated learning with {self.epochs} epochs")
        if self.quantization_enabled:
            logger.info("Quantization simulation enabled")
        
        for epoch in range(self.epochs):
            logger.info(f"=== Epoch: {epoch} ===")
            client_ids = [i for i in range(self.num_clients)]
            client_weight_datas = self.distribute_task(client_ids)
            self.federated_average(client_weight_datas)
            self.evalute()
            
            # 根据准确率动态调整稀疏度，更保守的策略
            current_sparsity = self.client.distill_args.target_sparsity
            
            # 如果准确率历史记录显示下降趋势，减缓稀疏度增加
            if len(self.accuracy_history) >= 2 and self.accuracy_history[-1] < self.accuracy_history[-2]:
                sparsity_reduction = 0.05  # 更小的调整步长
                logger.info(f"Accuracy decreased, using smaller sparsity adjustment: {sparsity_reduction}")
            else:
                sparsity_reduction = 0.1  # 标准调整步长
            
            new_sparsity = max(0.1, current_sparsity - sparsity_reduction)
            self.client.distill_args.target_sparsity = new_sparsity
            
            logger.info(f"Updated target sparsity: {current_sparsity:.2f} -> {new_sparsity:.2f}")
            
            # 如果准确率持续很低，提前停止训练
            if len(self.accuracy_history) >= 5 and all(acc < 0.55 for acc in self.accuracy_history[-5:]):
                logger.warning("🚨 Accuracy consistently very low for 5 epochs. Consider stopping training.")
                logger.warning("Suggestions: 1) Reduce learning rate 2) Reduce target sparsity 3) Check data quality")
