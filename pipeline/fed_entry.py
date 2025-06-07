import numpy as np
import logging
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    DataCollatorWithPadding
)
from transformers import Trainer
from transformers.models.bert.modeling_bert import BertForSequenceClassification as TModel

from modeling.modeling_cofi_bert import CoFiBertForSequenceClassification as SModel
from .args import (
    TrainingArguments,
    ModelArguments,
)
from .trainer import DistillTrainer

# ========== DP模块导入 ==========
try:
    from .dp_core import (
        DPConfig, DifferentialPrivacyManager,
        FederatedDPTraining, create_dp_federated_trainer
    )

    DP_AVAILABLE = True
    logging.info("✅ Differential Privacy modules loaded successfully!")
except ImportError as e:
    DP_AVAILABLE = False
    logging.warning(f"❌ DP import failed: {e}")
    logging.warning("Differential Privacy features disabled.")
    logging.warning("To enable DP, ensure dp_core.py is available in the same directory.")
except Exception as e:
    DP_AVAILABLE = False
    logging.error(f"❌ DP module error: {e}")
    logging.error("Differential Privacy features disabled due to module error.")

try:
    from .quantization_utils import (
        # 使用实际存在的函数名
        quantize_state_dict_real,
        dequantize_state_dict_real,
        serialize_quantized_weights,
        deserialize_quantized_weights,
        measure_real_communication_savings,
        get_calibration_dataloader,
        simulate_quantization_communication_cost,  # 保持兼容性
        get_model_size_info,
        QuantizationConfig
    )

    QUANTIZATION_AVAILABLE = True
    logging.info("✅ Quantization utilities loaded successfully!")
except ImportError as e:
    QUANTIZATION_AVAILABLE = False
    logging.warning(f"❌ Quantization import failed: {e}")
    logging.warning("Quantization features disabled.")

from copy import deepcopy

Trainers = Union[Trainer, DistillTrainer]

# Set up logging
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
        self.training_args = training_args  # 保存training_args引用

        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 1
        self.distill_args.gradient_accumulation_steps = 4

        # ========== DP初始化 ==========
        self.dp_trainer = None
        if hasattr(training_args, 'apply_dp') and training_args.apply_dp:
            if not DP_AVAILABLE:
                error_msg = "DP requested but dp_core module not available. Please ensure dp_core.py is in the pipeline directory."
                logger.error(f"❌ {error_msg}")
                raise ImportError(error_msg)

            try:
                logger.info("🔒 Initializing Differential Privacy for client...")
                # DP将在Server中统一管理，这里只是标记
                self.dp_enabled = True
                logger.info("✅ Client DP mode enabled")
            except Exception as e:
                logger.error(f"❌ Client DP initialization failed: {e}")
                self.dp_enabled = False
        else:
            self.dp_enabled = False
            logger.info("ℹ️  Client DP mode disabled")

        # Initialize quantization config
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
                    self.quantization_config.validate()
                    logger.info(f"Quantization enabled: {self.quantization_config.quantization_type}")
            except Exception as e:
                logger.warning(f"Quantization initialization failed: {e}")
                self.quantization_config = None

    def set_dp_trainer(self, dp_trainer):
        """设置DP训练器（由Server调用）"""
        self.dp_trainer = dp_trainer
        logger.info("🔒 DP trainer set for client")

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
        """Creates a safe state dictionary, ensuring all parameters are leaf nodes and numerically stable."""
        safe_state_dict = {}
        for key, param in model.state_dict().items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.warning(f"Parameter {key} contains NaN or Inf values, skipping client update")
                return None

            if param.requires_grad:
                safe_param = param.detach().clone()
                if safe_param.abs().max() > 100:
                    logger.warning(f"Parameter {key} has large values (max: {safe_param.abs().max()}), clipping")
                    safe_param = torch.clamp(safe_param, min=-10, max=10)
                safe_param.requires_grad_(True)
                safe_state_dict[key] = safe_param
            else:
                safe_state_dict[key] = param.detach().clone()
        return safe_state_dict

    def _create_client_trainer(self, server_model, t_model, client_id):
        """为客户端创建训练器实例"""
        try:
            # 创建训练参数副本
            client_train_args = deepcopy(self.distill_args)
            client_train_args.output_dir = f"./client_{client_id}_output"

            # 获取客户端数据集
            train_dataset = self.client_train_datas[client_id]
            eval_dataset = self.dataset.get('validation', None)

            # 确保模型处于正确状态
            server_model.train()

            # 创建训练器
            trainer = DistillTrainer(
                server_model,
                t_model,
                args=client_train_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )

            logger.info(f"✅ Client {client_id}: Trainer created successfully")
            return trainer

        except Exception as e:
            logger.error(f"❌ Client {client_id}: Failed to create trainer: {e}")
            logger.exception("Full trainer creation error:")
            return None

    def _create_fallback_update(self, server_weights):
        """创建后备更新（当训练失败时）"""
        try:
            if self.dp_enabled:
                # DP模式：返回零增量
                logger.info("Creating zero update for DP fallback")
                return {key: torch.zeros_like(tensor) for key, tensor in server_weights.items()}
            else:
                # 非DP模式：返回原权重
                logger.info("Returning original weights for non-DP fallback")
                return deepcopy(server_weights)
        except Exception as e:
            logger.error(f"Failed to create fallback update: {e}")
            # 最后的安全回退
            if isinstance(server_weights, dict):
                return {key: torch.zeros_like(tensor) if torch.is_tensor(tensor) else tensor
                        for key, tensor in server_weights.items()}
            else:
                return server_weights

    def _apply_quantization_simulation(self, server_model, client_id, datasets):
        """Performs safe quantization simulation for analysis only, without affecting training."""
        if not self.quantization_config or not self.quantization_config.apply_quantization:
            return

        try:
            logger.info(f"Client {client_id}: Running quantization simulation (analysis only)...")
            current_state_dict = server_model.state_dict()

            if QUANTIZATION_AVAILABLE:
                comm_stats = simulate_quantization_communication_cost(current_state_dict)
                logger.info(
                    f"Client {client_id}: Simulated communication reduction: {comm_stats['compression_ratio']:.2f}x")
                try:
                    model_info = get_model_size_info(server_model)
                    logger.info(f"Client {client_id}: Model size: {model_info['model_size_mb']:.2f} MB")
                except:
                    pass
        except Exception as e:
            logger.warning(f"Client {client_id}: Quantization simulation failed: {e}")

    def train_epoch(self, server_model, client_id, server_weights, t_model):
        """
        ========== 修改后的训练方法，支持DP ==========
        """
        datasets = self.client_train_datas[client_id]

        if self.dp_enabled and self.dp_trainer:
            # ========== DP模式：使用完整的DP训练流程 ==========
            try:
                logger.info(f"🔒 Client {client_id}: Starting DP training")

                # 先加载服务器权重到模型
                try:
                    if (isinstance(server_weights, dict) and QUANTIZATION_AVAILABLE and
                            any(isinstance(v, dict) and v.get('quantized', False) for v in server_weights.values())):
                        logger.info(f"Client {client_id}: Dequantizing received weights...")
                        server_weights = dequantize_state_dict_real(server_weights)

                    server_model.load_state_dict(server_weights, strict=True)
                    logger.info(f"✅ Client {client_id}: Server weights loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️  Client {client_id}: Failed to load weights strictly: {e}")
                    try:
                        server_model.load_state_dict(server_weights, strict=False)
                        logger.info(f"✅ Client {client_id}: Server weights loaded with strict=False")
                    except Exception as e2:
                        logger.error(f"❌ Client {client_id}: Failed to load weights at all: {e2}")
                        return self._create_fallback_update(server_weights)

                # 创建客户端训练器
                client_trainer = self._create_client_trainer(server_model, t_model, client_id)

                if client_trainer is None:
                    logger.error(f"❌ Client {client_id}: Failed to create trainer, falling back to non-DP training")
                    return self._original_train_epoch(server_model, client_id, server_weights, t_model, datasets)

                # 使用DP训练步骤
                clipped_update = self.dp_trainer.client_training_step(
                    client_trainer=client_trainer,
                    server_weights=server_weights,
                    client_train_data=datasets,
                    client_id=client_id,
                    training_args=self.training_args,
                    teacher_model=t_model
                )

                logger.info(f"🔒 Client {client_id}: DP training completed, returning clipped update")
                return clipped_update

            except Exception as e:
                logger.error(f"❌ Client {client_id}: DP training failed: {e}")
                logger.exception("Full DP training error traceback:")
                logger.info(f"🔄 Client {client_id}: Falling back to standard training")
                return self._original_train_epoch(server_model, client_id, server_weights, t_model, datasets)

        else:
            # ========== 非DP模式：原有训练逻辑 ==========
            return self._original_train_epoch(server_model, client_id, server_weights, t_model, datasets)

    def _original_train_epoch(self, server_model, client_id, server_weights, t_model, datasets):
        """原有的训练逻辑"""

        # Safely load weights
        try:
            # If server_weights are quantized, dequantize them first
            if isinstance(server_weights, dict) and any(isinstance(v, dict) and v.get('quantized', False)
                                                        for v in server_weights.values()):
                logger.info(f"Client {client_id}: Dequantizing received weights...")
                server_weights = dequantize_state_dict_real(server_weights)

            server_model.load_state_dict(server_weights, strict=True)
        except Exception as e:
            logger.warning(f"Client {client_id}: Failed to load weights strictly: {e}")
            server_model.load_state_dict(server_weights, strict=False)

        # Create trainer and execute training
        conservative_distill_args = deepcopy(self.distill_args)
        if hasattr(conservative_distill_args, 'learning_rate') and conservative_distill_args.learning_rate > 1e-4:
            conservative_distill_args.learning_rate = min(conservative_distill_args.learning_rate, 1e-4)
        if hasattr(conservative_distill_args,
                   'distill_learning_rate') and conservative_distill_args.distill_learning_rate > 1e-4:
            conservative_distill_args.distill_learning_rate = min(conservative_distill_args.distill_learning_rate, 1e-4)
        if hasattr(conservative_distill_args, 'target_sparsity') and conservative_distill_args.target_sparsity > 0.9:
            conservative_distill_args.target_sparsity = 0.8

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
            # Execute training
            server_model.train()
            distill_trainer.train()
            server_model.eval()

            # Post-training parameter check
            post_training_norms = []
            for name, param in server_model.named_parameters():
                if param.requires_grad:
                    param_norm = param.norm().item()
                    post_training_norms.append(param_norm)
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logger.error(f"Client {client_id}: Parameter {name} contains NaN/Inf after training")
                        return server_weights

            avg_post_norm = np.mean(post_training_norms) if post_training_norms else 0
            logger.info(f"Client {client_id}: Average parameter norm after training: {avg_post_norm:.4f}")

            # === Key Change: Real quantization implementation ===
            trained_weights = self._create_safe_state_dict(server_model)
            if trained_weights is None:
                logger.error(f"Client {client_id}: Unsafe weights detected, returning original weights")
                return server_weights

            # 🔧 Apply real quantization
            if (self.quantization_config and
                    self.quantization_config.apply_quantization and
                    QUANTIZATION_AVAILABLE):

                logger.info(f"Client {client_id}: Applying real quantization to trained weights...")

                # Perform real quantization
                quantized_weights, compression_stats = quantize_state_dict_real(
                    trained_weights, self.quantization_config
                )

                # Measure real communication savings
                comm_savings = measure_real_communication_savings(trained_weights, quantized_weights)

                logger.info(f"Client {client_id}: Real quantization results:")
                logger.info(f"   📊 Compression: {compression_stats['compression_ratio']:.2f}x")
                logger.info(
                    f"   💾 Size reduction: {compression_stats['original_size_mb']:.2f}MB -> {compression_stats['quantized_size_mb']:.2f}MB")
                logger.info(f"   📡 Communication savings: {comm_savings['savings_percentage']:.1f}%")
                logger.info(
                    f"   🔢 Quantized params: {compression_stats['quantized_params']}/{compression_stats['total_params']}")

                # Return quantized or dequantized weights based on strategy
                if hasattr(self.quantization_config, 'client_quantization_strategy'):
                    if self.quantization_config.client_quantization_strategy == "send_quantized":
                        logger.info(f"Client {client_id}: Sending quantized weights")
                        return quantized_weights
                    else:
                        logger.info(f"Client {client_id}: Dequantizing weights before sending")
                        dequantized_weights = dequantize_state_dict_real(quantized_weights)
                        return dequantized_weights
                else:
                    # Default: dequantize before sending
                    dequantized_weights = dequantize_state_dict_real(quantized_weights)
                    return dequantized_weights
            else:
                # No quantization case
                logger.info(f"Client {client_id}: Training completed without quantization")
                return trained_weights

        except Exception as e:
            logger.error(f"Client {client_id}: Training failed: {e}")
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

        # ========== DP初始化 ==========
        self.dp_trainer = None
        if hasattr(training_args, 'apply_dp') and training_args.apply_dp:
            if not DP_AVAILABLE:
                error_msg = "DP requested but dp_core module not available. Please ensure dp_core.py is in the pipeline directory."
                logger.error(f"❌ {error_msg}")
                if hasattr(training_args, 'dp_strict_mode') and training_args.dp_strict_mode:
                    raise ImportError(error_msg)
                else:
                    logger.warning("⚠️  Continuing without DP protection (non-strict mode)")
                    return

            try:
                logger.info("🔒 Initializing Differential Privacy for server...")

                # 创建DP训练器
                self.dp_trainer = create_dp_federated_trainer(training_args)

                if self.dp_trainer is None:
                    logger.error("❌ Failed to create DP trainer")
                    raise ValueError("DP trainer creation failed")

                # 将DP训练器传递给客户端
                self.client.set_dp_trainer(self.dp_trainer)

                logger.info(f"✅ DP-FedAvg enabled:")
                logger.info(f"   🎯 Target ε: {training_args.dp_target_epsilon}")
                logger.info(f"   🔊 Noise multiplier: {training_args.dp_noise_multiplier}")
                logger.info(f"   ✂️  Clipping bound: {training_args.dp_clipping_bound}")
                logger.info(f"   📊 Accountant: {training_args.dp_accountant_type}")

                # 验证DP trainer组件
                if not hasattr(self.dp_trainer, 'dp_manager'):
                    logger.error("❌ DP trainer missing dp_manager")
                    raise ValueError("Invalid DP trainer structure")

                if not hasattr(self.dp_trainer, 'client_training_step'):
                    logger.error("❌ DP trainer missing client_training_step method")
                    raise ValueError("Invalid DP trainer methods")

                logger.info("✅ DP trainer validation passed")

            except Exception as e:
                logger.error(f"❌ DP initialization failed: {e}")
                logger.exception("Full DP initialization error:")
                self.dp_trainer = None
                # 可以选择是否要抛出异常或继续运行
                if hasattr(training_args, 'dp_strict_mode') and training_args.dp_strict_mode:
                    raise RuntimeError(f"DP initialization failed in strict mode: {e}")
                else:
                    logger.warning("⚠️  Continuing without DP protection")
        else:
            logger.info("ℹ️  Standard federated learning (no DP)")

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

        self.accuracy_history = []
        self.accuracy_drop_threshold = 0.1

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

        # 🔧 修复7: 正确的剪枝调度策略
        # 原问题：稀疏度策略逻辑反向，在减少而非增加稀疏度
        self.initial_target_sparsity = 0.1  # 起始稀疏度：10%
        self.final_target_sparsity = 0.8  # 最终稀疏度：80%
        self.sparsity_schedule = 'gradual'  # 调度策略：渐进式
        self.sparsity_patience = 3  # 稀疏度调整的耐心值
        self.consecutive_drops = 0  # 连续下降次数

        # 新增：稀疏度增长控制参数
        self.max_sparsity_increase_per_epoch = 0.05  # 每轮最大稀疏度增长
        self.sparsity_adjustment_factor = 0.8  # 稀疏度调整因子
        self.min_accuracy_threshold = 0.6  # 最低准确率阈值

        logger.info(f"🎯 Pruning schedule: {self.initial_target_sparsity:.1f} → {self.final_target_sparsity:.1f}")
        logger.info(f"📈 Max sparsity increase per epoch: {self.max_sparsity_increase_per_epoch:.2f}")

    def _validate_client_update(self, update, reference_weights):
        """验证客户端更新的有效性"""
        if not isinstance(update, dict):
            return False

        # 检查键匹配
        if set(update.keys()) != set(reference_weights.keys()):
            logger.warning("Client update keys don't match server weights")
            return False

        # 检查张量形状
        for key in update:
            if update[key].shape != reference_weights[key].shape:
                logger.warning(f"Shape mismatch for {key}")
                return False

        return True

    def distribute_task(self, client_ids):
        """
        ========== 修改后的任务分发，支持DP ==========
        """
        logger.info(f"Distributing task to {len(client_ids)} clients")

        server_weights = deepcopy(self.s_model.state_dict())
        client_updates = []

        for i in range(len(client_ids)):
            client_id = client_ids[i]
            logger.info(f"Training client {client_id}")

            try:
                if self.dp_trainer:
                    # ========== DP模式：客户端返回剪裁后的更新增量 ==========
                    update = self.client.train_epoch(self.s_model, client_id, server_weights, self.t_model)

                    # 验证更新格式
                    if self._validate_client_update(update, server_weights):
                        client_updates.append(update)
                        logger.info(f"🔒 Client {client_id}: Valid DP update received")
                    else:
                        logger.warning(f"❌ Client {client_id}: Invalid DP update, skipping")

                else:
                    # ========== 非DP模式：客户端返回完整权重 ==========
                    weight = self.client.train_epoch(self.s_model, client_id, server_weights, self.t_model)
                    client_updates.append(weight)

            except Exception as e:
                logger.error(f"❌ Client {client_id} training failed: {e}")
                continue

        logger.info(f"Collected updates from {len(client_updates)} clients")
        return client_updates

    def federated_average(self, client_weight_datas):
        """
        ========== 修改后的联邦平均，支持DP ==========
        """
        if not client_weight_datas:
            logger.error("No client data for aggregation!")
            return self.s_model.state_dict()

        if self.dp_trainer:
            # ========== DP模式：使用DP聚合 ==========
            logger.info(f"🔒 Performing DP aggregation")

            # 检查隐私预算
            if not self.dp_trainer.dp_manager.can_continue_training():
                current_eps, _ = self.dp_trainer.dp_manager.get_privacy_budget()
                logger.error(f"🛑 Privacy budget exhausted! ε={current_eps:.4f}")
                return self.s_model.state_dict()

            # DP聚合（包含噪声添加）
            try:
                # 确定当前轮次
                current_round = len(self.accuracy_history)

                aggregated_update = self.dp_trainer.server_aggregation_step(
                    client_weight_datas, current_round
                )

                if aggregated_update:
                    # 应用聚合更新到服务器模型
                    server_lr = getattr(self.training_args, 'server_learning_rate', 1.0)
                    self.dp_trainer.apply_update_to_server(
                        self.s_model,
                        aggregated_update,
                        learning_rate=server_lr
                    )

                    # 输出隐私状态
                    privacy_status = self.dp_trainer.get_privacy_status()
                    logger.info(f"🔒 Privacy status:")
                    logger.info(f"   ε: {privacy_status['current_epsilon']:.4f}/{privacy_status['target_epsilon']:.4f}")
                    logger.info(f"   Budget used: {privacy_status['budget_utilization']:.1%}")
                    logger.info(f"   Remaining: {privacy_status['remaining_budget']:.4f}")

                return self.s_model.state_dict()

            except Exception as e:
                logger.error(f"❌ DP aggregation failed: {e}")
                return self.s_model.state_dict()

        else:
            # ========== 非DP模式：原有联邦平均 ==========
            return self._original_federated_average(client_weight_datas)

    def _original_federated_average(self, client_weight_datas):
        """原有的联邦平均实现"""
        # Validate and process client weights
        valid_client_weights = self._validate_and_process_client_weights(client_weight_datas)
        client_num = len(valid_client_weights)

        if client_num == 0:
            logger.error("No valid client weights for aggregation!")
            return self.s_model.state_dict()

        logger.info(f"Server: Starting federated aggregation with {client_num} clients...")

        # 🔧 Handle aggregation of quantized weights
        if self._has_quantized_weights(valid_client_weights):
            logger.info("Server: Detected quantized weights from clients, dequantizing for aggregation...")
            dequantized_weights = []
            for i, weights in enumerate(valid_client_weights):
                if self._is_quantized_weights(weights):
                    deq_weights = dequantize_state_dict_real(weights)
                    dequantized_weights.append(deq_weights)
                    logger.info(f"Server: Dequantized weights from client {i}")
                else:
                    dequantized_weights.append(weights)
            valid_client_weights = dequantized_weights

        # Perform standard federated averaging
        aggregated_w = self._perform_federated_averaging(valid_client_weights)

        # Apply server-side noise if needed
        final_w = self._apply_server_noise(aggregated_w)

        # 🔧 Server-side quantization (if enabled)
        if (self.quantization_enabled and
                hasattr(self.training_args, 'quantize_global_model') and
                self.training_args.quantize_global_model and
                QUANTIZATION_AVAILABLE):

            logger.info("Server: Applying quantization to global model...")
            quantized_global, global_stats = quantize_state_dict_real(
                final_w,
                type('Config', (), {'apply_quantization': True})()
            )

            logger.info("Server: Global model quantization results:")
            logger.info(f"   📊 Compression: {global_stats['compression_ratio']:.2f}x")
            logger.info(
                f"   💾 Size: {global_stats['original_size_mb']:.2f}MB -> {global_stats['quantized_size_mb']:.2f}MB")

            self.s_model.load_state_dict(dequantize_state_dict_real(quantized_global))
            return final_w
        else:
            # Standard case: load and return weights
            try:
                self.s_model.load_state_dict(final_w, strict=True)
                logger.info("Server: Model weights updated successfully")
            except RuntimeError as e:
                logger.error(f"Failed to load state_dict: {e}")
                self.s_model.load_state_dict(final_w, strict=False)

        return final_w

    def _validate_and_process_client_weights(self, client_weight_datas):
        """Validate and process client weights."""
        valid_weights = []
        for i, weights in enumerate(client_weight_datas):
            if self._is_quantized_weights(weights):
                if self._validate_quantized_weights(weights):
                    valid_weights.append(weights)
                else:
                    logger.warning(f"Client {i} quantized weights are invalid, excluding from aggregation")
            else:
                is_valid = True
                for key, param in weights.items():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logger.warning(f"Client {i} weights contain NaN/Inf in {key}, excluding from aggregation")
                        is_valid = False
                        break
                if is_valid:
                    valid_weights.append(weights)

        if len(valid_weights) == 0:
            logger.error("No valid client weights for aggregation!")
            return client_weight_datas

        logger.info(f"Using {len(valid_weights)}/{len(client_weight_datas)} client weights for aggregation")
        return valid_weights

    def _has_quantized_weights(self, client_weights_list):
        """Check if any weights are quantized."""
        return any(self._is_quantized_weights(weights) for weights in client_weights_list)

    def _is_quantized_weights(self, weights):
        """Check if a set of weights is quantized."""
        if not isinstance(weights, dict):
            return False
        return any(isinstance(v, dict) and v.get('quantized', False) for v in weights.values())

    def _validate_quantized_weights(self, quantized_weights):
        """Validate the integrity of quantized weights."""
        try:
            for key, value in quantized_weights.items():
                if isinstance(value, dict) and value.get('quantized', False):
                    if 'data' not in value or 'scale' not in value or 'zero_point' not in value:
                        return False
                    if torch.isnan(torch.tensor(value['scale'])) or torch.isinf(torch.tensor(value['scale'])):
                        return False
            return True
        except:
            return False

    def _perform_federated_averaging(self, valid_client_weights):
        """Perform federated averaging."""
        client_num = len(valid_client_weights)
        first_client_w = valid_client_weights[0]
        aggregated_w = OrderedDict()

        for key in first_client_w.keys():
            param_template = first_client_w[key]

            if param_template.is_floating_point():
                sum_param = torch.zeros_like(param_template)
                valid_param_count = 0

                for i in range(client_num):
                    client_param = valid_client_weights[i][key]
                    if not (torch.isnan(client_param).any() or torch.isinf(client_param).any()):
                        sum_param += client_param.to(sum_param.device)
                        valid_param_count += 1

                if valid_param_count > 0:
                    aggregated_w[key] = sum_param / valid_param_count
                else:
                    logger.warning(f"All client weights for {key} are invalid, keeping server weight")
                    aggregated_w[key] = self.s_model.state_dict()[key].clone()
            else:
                aggregated_w[key] = param_template.clone()

        return aggregated_w

    def _apply_server_noise(self, aggregated_w):
        """Apply server-side noise."""
        if (hasattr(self.training_args, 'global_noise_type') and
                self.training_args.global_noise_type.lower() == 'laplace' and
                hasattr(self.training_args, 'global_noise_scale') and
                self.training_args.global_noise_scale > 0):

            # 如果启用了DP，警告用户不要同时使用两种噪声机制
            if self.dp_trainer:
                logger.warning("⚠️  Both DP and legacy Laplacian noise enabled! Using DP only.")
                return aggregated_w

            reduced_noise_scale = min(self.training_args.global_noise_scale, 0.01)
            final_w = add_laplacian_noise_to_state_dict(aggregated_w, reduced_noise_scale)
            logger.info(f"Applied reduced noise scale: {reduced_noise_scale}")
            return final_w
        return aggregated_w

    def compute_metrics(self, eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return {"accuracy": accuracy}

    def _check_accuracy_drop(self, current_accuracy):
        """Checks for abnormal drops in accuracy."""
        self.accuracy_history.append(current_accuracy)

        if len(self.accuracy_history) >= 3:
            recent_accuracies = self.accuracy_history[-3:]
            if all(acc < self.min_accuracy_threshold for acc in recent_accuracies):
                logger.warning(
                    f"🚨 Accuracy consistently below {self.min_accuracy_threshold}! Possible training instability.")
                return True
            if len(self.accuracy_history) >= 2:
                prev_acc = self.accuracy_history[-2]
                if prev_acc - current_accuracy > self.accuracy_drop_threshold:
                    logger.warning(f"🚨 Significant accuracy drop: {prev_acc:.4f} -> {current_accuracy:.4f}")
                    return True

        return False

    def _log_dp_progress(self, epoch):
        """记录DP训练进度"""
        if not self.dp_trainer:
            return

        privacy_status = self.dp_trainer.get_privacy_status()
        stats = self.dp_trainer.dp_manager.get_statistics()

        logger.info(f"📊 DP Progress [Epoch {epoch + 1}]:")
        logger.info(f"   🔒 Privacy: ε={privacy_status['current_epsilon']:.4f}/"
                    f"{privacy_status['target_epsilon']:.4f} "
                    f"({privacy_status['budget_utilization']:.1%})")
        logger.info(f"   ✂️  Clipping: {stats['clipping_rate']:.1%} of updates clipped")
        logger.info(f"   📉 Remaining budget: {privacy_status['remaining_budget']:.4f}")

    def _log_final_dp_summary(self):
        """记录最终DP摘要"""
        if not self.dp_trainer:
            return

        training_summary = self.dp_trainer.get_training_summary()
        privacy_status = self.dp_trainer.get_privacy_status()

        logger.info("🎉 === DP TRAINING COMPLETED ===")
        logger.info(f"🔒 Final Privacy Consumption:")
        logger.info(f"   ε = {privacy_status['current_epsilon']:.4f} / {privacy_status['target_epsilon']:.4f}")
        logger.info(f"   δ = {privacy_status['delta']}")
        logger.info(f"   Budget utilization: {privacy_status['budget_utilization']:.1%}")
        logger.info(f"📈 Training Statistics:")
        logger.info(f"   Total rounds: {training_summary['total_rounds']}")
        logger.info(f"   Total clients trained: {training_summary['total_clients_trained']}")
        logger.info(f"   Avg clients per round: {training_summary['avg_clients_per_round']:.1f}")

    def evalute(self):
        # Check if model weights are normal before evaluation
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

            accuracy_drop_detected = self._check_accuracy_drop(current_accuracy)
            if accuracy_drop_detected:
                logger.warning("Consider reducing learning rate or target sparsity")

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
            try:
                self._simple_accuracy_check()
            except:
                logger.error("Even simple accuracy check failed!")

    def _simple_accuracy_check(self):
        """A simple accuracy check as a fallback for evaluation failures."""
        logger.info("Running simple accuracy check...")
        self.s_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, sample in enumerate(self.dataset):
                if i >= 100:
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

    def adaptive_sparsity_scheduling(self, current_epoch, current_accuracy):
        """
        🔧 修复8: 完全重写稀疏度调度逻辑，解决原来的反向逻辑问题
        原问题：代码在减少稀疏度而非增加
        修正：正确实现从低稀疏度到高稀疏度的渐进剪枝
        """
        # 获取当前稀疏度目标
        current_target_sparsity = self.client.distill_args.target_sparsity

        # 🔧 核心修复：正确的稀疏度增长逻辑
        # 基于epoch的基础调度
        if self.sparsity_schedule == 'gradual':
            # 线性增长：从initial_target_sparsity到final_target_sparsity
            progress = min(current_epoch / (self.epochs * 0.8), 1.0)  # 在80%的训练时间内完成剪枝
            base_target_sparsity = self.initial_target_sparsity + progress * (
                    self.final_target_sparsity - self.initial_target_sparsity
            )
        elif self.sparsity_schedule == 'aggressive':
            # 更激进的调度
            progress = min(current_epoch / (self.epochs * 0.6), 1.0)  # 在60%的训练时间内完成剪枝
            base_target_sparsity = self.initial_target_sparsity + progress * (
                    self.final_target_sparsity - self.initial_target_sparsity
            )
        else:
            # 固定稀疏度
            base_target_sparsity = self.final_target_sparsity

        # 🔧 自适应调整：基于准确率变化
        new_target_sparsity = base_target_sparsity

        if len(self.accuracy_history) >= 2:
            recent_acc = self.accuracy_history[-1]
            prev_acc = self.accuracy_history[-2]

            if recent_acc < prev_acc:
                # 准确率下降，减缓剪枝速度
                self.consecutive_drops += 1
                logger.warning(
                    f"📉 Accuracy dropped: {prev_acc:.4f} → {recent_acc:.4f} (consecutive drops: {self.consecutive_drops})")

                if self.consecutive_drops >= self.sparsity_patience:
                    # 多次连续下降，显著减缓剪枝
                    sparsity_increase = base_target_sparsity - current_target_sparsity
                    adjusted_increase = sparsity_increase * self.sparsity_adjustment_factor
                    new_target_sparsity = current_target_sparsity + adjusted_increase

                    logger.warning(f"🚨 Slowing down pruning due to {self.consecutive_drops} consecutive accuracy drops")
                    logger.info(f"🔧 Reduced sparsity increase: {sparsity_increase:.4f} → {adjusted_increase:.4f}")
                else:
                    # 单次下降，轻微减缓
                    sparsity_increase = base_target_sparsity - current_target_sparsity
                    adjusted_increase = sparsity_increase * 0.7  # 减少30%的增长
                    new_target_sparsity = current_target_sparsity + adjusted_increase
            else:
                # 准确率稳定或上升，正常调度
                self.consecutive_drops = 0
                new_target_sparsity = base_target_sparsity

        # 🔧 关键修复：确保稀疏度单调递增且有界
        # 1. 不能低于当前稀疏度（单调递增）
        new_target_sparsity = max(new_target_sparsity, current_target_sparsity)

        # 2. 限制每轮的最大增长
        max_allowed_sparsity = current_target_sparsity + self.max_sparsity_increase_per_epoch
        new_target_sparsity = min(new_target_sparsity, max_allowed_sparsity)

        # 3. 不能超过最终目标
        new_target_sparsity = min(new_target_sparsity, self.final_target_sparsity)

        # 4. 确保在合理范围内
        new_target_sparsity = max(new_target_sparsity, 0.0)
        new_target_sparsity = min(new_target_sparsity, 0.95)  # 最多剪枝95%

        # 应用新的稀疏度目标
        if abs(new_target_sparsity - current_target_sparsity) > 0.001:  # 避免无意义的微小更新
            sparsity_increase = new_target_sparsity - current_target_sparsity
            progress_percentage = (new_target_sparsity - self.initial_target_sparsity) / (
                    self.final_target_sparsity - self.initial_target_sparsity) * 100

            logger.info(f"🎯 Sparsity update: {current_target_sparsity:.4f} → {new_target_sparsity:.4f} "
                        f"(+{sparsity_increase:.4f})")
            logger.info(f"📊 Pruning progress: {progress_percentage:.1f}% toward final target")

            # 更新客户端的目标稀疏度
            self.client.distill_args.target_sparsity = new_target_sparsity

            # 🔧 新增：直接更新模型掩码
            try:
                # 创建一个临时trainer来访问掩码更新方法
                temp_trainer = DistillTrainer(
                    self.s_model,
                    self.t_model,
                    args=self.distill_args,
                    eval_dataset=self.dataset,
                    tokenizer=self.tokenizer,
                    data_collator=self.data_collator,
                    compute_metrics=self.compute_metrics,
                )
                temp_trainer.update_masks_for_target_sparsity(new_target_sparsity)
                logger.info("✅ Updated model masks with new target sparsity")
            except Exception as e:
                logger.warning(f"Failed to update masks directly: {e}")

        return new_target_sparsity

    def run(self):
        """
        ========== 修改后的训练主循环，支持DP监控 ==========
        """
        logger.info(f"Starting federated learning with {self.epochs} epochs")
        logger.info(f"🎯 Pruning schedule: {self.initial_target_sparsity:.1f} → {self.final_target_sparsity:.1f}")
        logger.info(
            f"📈 Strategy: {self.sparsity_schedule}, Max increase/epoch: {self.max_sparsity_increase_per_epoch:.2f}")

        # DP状态显示
        if self.dp_trainer:
            privacy_status = self.dp_trainer.get_privacy_status()
            logger.info(f"🔒 DP Protection:")
            logger.info(f"   Target ε: {privacy_status['target_epsilon']}")
            logger.info(f"   Clipping bound: {self.training_args.dp_clipping_bound}")
            logger.info(f"   Noise multiplier: {self.training_args.dp_noise_multiplier}")

        # 设置初始稀疏度
        self.client.distill_args.target_sparsity = self.initial_target_sparsity

        for epoch in range(self.epochs):
            logger.info(f"=== Epoch: {epoch + 1}/{self.epochs} ===")

            # ========== DP预算检查 ==========
            if self.dp_trainer and not self.dp_trainer.dp_manager.can_continue_training():
                current_eps, _ = self.dp_trainer.dp_manager.get_privacy_budget()
                logger.error(f"🛑 Training stopped: Privacy budget exhausted at epoch {epoch + 1}!")
                logger.error(f"Final ε = {current_eps:.4f} > target {self.training_args.dp_target_epsilon}")
                break

            # 显示当前剪枝状态
            current_target = self.client.distill_args.target_sparsity
            progress = (current_target - self.initial_target_sparsity) / (
                    self.final_target_sparsity - self.initial_target_sparsity) * 100
            logger.info(f"🎯 Current target sparsity: {current_target:.4f} ({current_target * 100:.1f}%)")
            logger.info(f"📊 Pruning progress: {progress:.1f}%")

            # 执行联邦训练
            client_ids = [i for i in range(self.num_clients)]
            client_weight_datas = self.distribute_task(client_ids)
            self.federated_average(client_weight_datas)

            # 评估模型
            self.evalute()

            # 🔧 使用修复后的自适应稀疏度调度
            current_accuracy = self.accuracy_history[-1] if self.accuracy_history else 0.0
            new_sparsity = self.adaptive_sparsity_scheduling(epoch, current_accuracy)

            # ========== DP统计输出 ==========
            if self.dp_trainer and epoch % 2 == 0:
                self._log_dp_progress(epoch)

            # 早停检查（准确率持续过低）
            if len(self.accuracy_history) >= 5:
                recent_accuracies = self.accuracy_history[-5:]
                if all(acc < self.min_accuracy_threshold for acc in recent_accuracies):
                    logger.error(
                        f"🛑 Training stopped: Accuracy below {self.min_accuracy_threshold} for 5 consecutive epochs")
                    logger.error("🔧 Recommendations:")
                    logger.error(f"   1. Reduce final target sparsity (current: {self.final_target_sparsity})")
                    logger.error(f"   2. Use more gradual sparsity schedule")
                    logger.error(f"   3. Increase sparsity patience (current: {self.sparsity_patience})")
                    logger.error(
                        f"   4. Reduce max sparsity increase per epoch (current: {self.max_sparsity_increase_per_epoch})")
                    break

            # 检查是否达到最终目标
            if abs(new_sparsity - self.final_target_sparsity) < 0.01:
                logger.info(f"🎉 Reached target sparsity: {new_sparsity:.4f}")

            logger.info("=" * 50)

        # ========== 训练完成总结 ==========
        logger.info("🎉 FEDERATED LEARNING COMPLETED")

        # 准确率总结
        if hasattr(self, 'accuracy_history') and self.accuracy_history:
            final_acc = self.accuracy_history[-1]
            best_acc = max(self.accuracy_history)
            logger.info(f"📊 Final accuracy: {final_acc:.4f}")
            logger.info(f"📊 Best accuracy: {best_acc:.4f}")

        # 剪枝总结
        final_sparsity = self.client.distill_args.target_sparsity
        achieved_progress = (final_sparsity - self.initial_target_sparsity) / (
                self.final_target_sparsity - self.initial_target_sparsity) * 100
        logger.info(f"🎯 Final target sparsity: {final_sparsity:.4f}")
        logger.info(f"📈 Pruning progress achieved: {achieved_progress:.1f}%")

        # ========== DP总结 ==========
        if self.dp_trainer:
            self._log_final_dp_summary()

        logger.info("=" * 50)
