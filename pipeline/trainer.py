import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EvalPrediction,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    DataCollator,
)
from typing import Dict, List, Any, Tuple, Callable, Union, Optional, Sequence
from tqdm import tqdm
from loguru import logger

# 配置日志系统，减少冗余输出
logger.remove()  # 移除默认处理程序
# 添加自定义格式的处理程序，使用不同颜色区分不同级别的日志
logger.add(
    lambda msg: tqdm.write(msg, end=""),  # 使用 tqdm.write 避免与进度条冲突
    format="<level>{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}</level>",
    level="INFO",
    colorize=True
)

from transformers import Trainer as DefaultTrainer
from transformers.trainer import (
    unwrap_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
)

from modeling.mask import Mask
from modeling.modeling_cofi_bert import (
    CoFiBertForSequenceClassification,
)

SModel = CoFiBertForSequenceClassification


class DistillTrainer(DefaultTrainer):

    def __init__(self,
                 s_model: Union[PreTrainedModel, nn.Module] = None,
                 t_model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
                 ):
        assert callbacks is None
        super().__init__(
            s_model, args, data_collator, train_dataset, eval_dataset,
            tokenizer, model_init, compute_metrics, callbacks, optimizers,
            preprocess_logits_for_metrics
        )
        self.t_model = t_model
        device = next(self.model.parameters()).device
        self.t_model.to(device)
        self.t_model.eval()

        self.distill_switch = False
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.mse_loss = nn.MSELoss()

        self.start_sparsity = 1.
        self.target_sparsity = self.args.target_sparsity

        # Initialize loss trackers for monitoring
        self.last_l_pred = 0.0
        self.last_l_layer = 0.0
        self.last_distill_loss = 0.0
        self.step_count = 0
        
        # 增加日志控制参数
        self.log_interval = 500  # 减少日志频率，每500步记录一次
        self.debug_mode = False  # 调试模式开关，默认关闭
        self.eval_log_once = False  # 确保评估时只记录一次详细信息

        self.reg_params = []

        self.per_layer_mask_groups: List[Tuple[Mask, ...]] = []
        self.init_reg_params()
        self.ffn_masks: List[Mask] = []
        self.init_ffn_masks()
        
        # 记录初始化完成
        logger.info("DistillTrainer initialized successfully")

    def init_ffn_masks(self):
        model: SModel = self.model
        for layer in model.bert.encoder.layer:
            FFN_mask = layer.output.mask
            self.ffn_masks.append(FFN_mask)

    def init_reg_params(self):
        for name, _ in self.model.named_parameters():
            if name.endswith('reg_lambda_1') or \
                    name.endswith('reg_lambda_2') or \
                    name.endswith('log_alpha'):
                self.reg_params.append(name)
        model: SModel = self.model

        for layer in model.bert.encoder.layer:
            head_mask = layer.attention.self.mask
            filter_mask = layer.output.dense.mask
            MHA_mask = layer.attention.output.mask
            FFN_mask = layer.output.mask

            self.per_layer_mask_groups.append((
                head_mask,
                MHA_mask,
                FFN_mask,
                filter_mask,
            ))

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n in decay_parameters and p.requires_grad and n not in self.reg_params)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n not in decay_parameters and p.requires_grad and n not in self.reg_params)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in self.reg_params and "reg" not in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.reg_learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in self.reg_params and "reg" in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": -self.args.reg_learning_rate,
                }
            ]

            optimizer_cls, optimizer_kwargs = DefaultTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def train(self,
              resume_from_checkpoint: Optional[Union[str, bool]] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None,
              ignore_keys_for_eval: Optional[List[str]] = None,
              **kwargs
              ):
        self.distill_switch = True
        # Reset loss tracking metrics at the start of training
        self.last_l_pred = 0.0
        self.last_l_layer = 0.0
        self.last_distill_loss = 0.0
        self.step_count = 0

        # 使用明确的训练开始横幅，便于在日志中查找
        logger.info("=" * 50)
        logger.info("TRAINING START")
        logger.info(f"Distill params: T={self.args.distill_T}, lambda={self.args.distill_lambda}")
        logger.info("=" * 50)

        result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self.distill_switch = False

        # 使用明确的训练结束横幅
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED")
        logger.info(f"Final losses - L_pred: {self.last_l_pred:.6f}, L_layer: {self.last_l_layer:.6f}, Combined: {self.last_distill_loss:.6f}")
        logger.info("=" * 50)

        return result

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if "output_hidden_states" in inputs:
            inputs["output_hidden_states"] = inputs["output_hidden_states"] or self.distill_switch
        else:
            inputs["output_hidden_states"] = self.distill_switch
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Distill Loss
        if self.distill_switch:
            distill_loss, l_pred, l_layer = self.compute_distill_loss(
                unwrap_model(model),
                inputs,
                outputs["logits"],
                outputs["hidden_states"],
                return_components=True
            )

            # Store the loss components for logging
            self.last_l_pred = l_pred.item()
            self.last_l_layer = l_layer.item()
            self.last_distill_loss = distill_loss.item()

            # 增量步数并按固定间隔记录日志，大幅减少日志频率
            self.step_count += 1
            if self.step_count % self.log_interval == 0:
                sparsity = self.compute_sparsity().item()
                logger.info(
                    f"[STEP {self.step_count}] L_pred: {self.last_l_pred:.4f}, L_layer: {self.last_l_layer:.4f}, "
                    f"Combined: {self.last_distill_loss:.4f}, Sparsity: {sparsity:.4f}"
                )

            if self.args.distill:
                loss = 0. * loss + distill_loss

        # Lagrangian Loss
        if self.distill_switch:
            lagrangian_loss = self.compute_lagrangian_loss()
            loss = loss + lagrangian_loss

        return (loss, outputs) if return_outputs else loss

    def mask_select(self,
                    value: torch.Tensor,
                    mask: torch.Tensor
                    ) -> torch.Tensor:
        assert value.shape[:-1] == mask.shape
        D = value.shape[-1]
        value = value.view(-1, D)
        mask = mask.view(-1).bool()
        return value[mask]

    def compute_distill_loss(self, model, inputs, s_logits, s_hidden_states, return_components=True):
        # 只在调试模式下记录详细信息
        if self.debug_mode:
            logger.debug(f"Computing distill loss - input keys: {inputs.keys()}")
            logger.debug(f"s_logits shape: {s_logits.shape}, s_hidden_states length: {len(s_hidden_states)}")
        
        # 确保输入包含 output_hidden_states 并设置为 True
        inputs_copy = inputs.copy()  # 创建副本避免修改原始输入
        inputs_copy["output_hidden_states"] = True
        
        with torch.no_grad():
            # 不再使用断言，而是确保参数正确
            t_outputs = self.t_model(**inputs_copy)
            t_logits = t_outputs["logits"]
            t_hidden_states = t_outputs["hidden_states"]
            
            if self.debug_mode:
                logger.debug(f"t_logits shape: {t_logits.shape}, t_hidden_states length: {len(t_hidden_states)}")
        
        if "attention_mask" not in inputs:
            if self.debug_mode:
                logger.debug("No attention_mask found in inputs, creating default mask")
            # 创建全1的注意力掩码
            attention_mask = torch.ones_like(inputs["input_ids"])
        else:
            attention_mask = inputs["attention_mask"]
            
        mask = attention_mask
        T = self.args.distill_T
        distill_lambda = self.args.distill_lambda

        # 计算预测损失
        pred_loss = self.kl_loss(
            torch.log_softmax(s_logits / T, dim=-1),
            torch.log_softmax(t_logits / T, dim=-1),
        )

        # 检查隐藏状态的长度是否匹配
        if len(t_hidden_states) != len(s_hidden_states):
            if self.debug_mode:
                logger.warning(f"Hidden states length mismatch: teacher={len(t_hidden_states)}, student={len(s_hidden_states)}")
            # 使用最小长度
            min_length = min(len(t_hidden_states), len(s_hidden_states))
            t_hidden_states = t_hidden_states[:min_length]
            s_hidden_states = s_hidden_states[:min_length]

        # 应用投影和掩码
        try:
            proj = model.bert.distill_projection
            t_hidden_states = [self.mask_select(t_h, mask) for t_h in t_hidden_states]
            s_hidden_states = [proj(self.mask_select(s_h, mask)) for s_h in s_hidden_states]
        except Exception as e:
            logger.error(f"Mask/projection error: {str(e)[:100]}...")  # 限制错误消息长度
            # 如果出错，尝试直接使用隐藏状态（不应用掩码）
            if self.debug_mode:
                logger.debug("Falling back to using hidden states without masking")
            t_mean_states = [t_h.mean(dim=1) for t_h in t_hidden_states]
            s_mean_states = [s_h.mean(dim=1) for s_h in s_hidden_states]
            
            # 计算简单的MSE损失
            layer_loss = torch.stack([self.mse_loss(t_h, s_h) 
                                     for t_h, s_h in zip(t_mean_states, s_mean_states)]).sum()
            
            # 组合损失
            distill_loss = distill_lambda * pred_loss + (1.0 - distill_lambda) * layer_loss
            
            if return_components:
                return distill_loss, pred_loss, layer_loss
            return distill_loss

        # 使用简单的层匹配策略
        match_index = []
        for i in range(len(s_hidden_states)):
            match_index.append(i)  # 直接匹配相同索引的层

        # 计算层损失
        try:
            _layer_loss = []
            for i, (ffn_mask, s_h) in enumerate(zip(self.ffn_masks, s_hidden_states)):
                t_h = t_hidden_states[match_index[i]]
                _layer_loss.append(self.mse_loss(t_h, s_h))
            layer_loss = torch.stack(_layer_loss).sum()
        except Exception as e:
            logger.error(f"Layer loss calculation error: {str(e)[:100]}...")  # 限制错误消息长度
            # 如果计算层损失出错，使用简化版本
            layer_loss = sum(self.mse_loss(t_hidden_states[i], s_hidden_states[i]) 
                            for i in range(len(s_hidden_states)))

        # 组合损失
        distill_loss = distill_lambda * pred_loss + (1.0 - distill_lambda) * layer_loss

        if return_components:
            return distill_loss, pred_loss, layer_loss
        return distill_loss

    def compute_target_sparsity(self):
        return self.target_sparsity

    def compute_lagrangian_loss(self):
        s = self.compute_sparsity()
        t = self.compute_target_sparsity()

        lambda_1 = self.model.bert.reg_lambda_1
        lambda_2 = self.model.bert.reg_lambda_2
        lagrangian_loss = lambda_1 * (s - t).abs() + lambda_2 * torch.pow(s - t, 2.)
        return lagrangian_loss

    def compute_sparsity(self):
        num_layers = 12
        num_heads = 12
        hidden_size = 768
        ffn_size = 768 * 4
        M = (hidden_size * hidden_size * 4 + hidden_size * ffn_size * 2) * num_layers
        params = []
        hidden_mask = torch.ones([768]).cuda()
        for mask_group in self.per_layer_mask_groups:
            head_mask, MHA_mask, FFN_mask, filter_mask = mask_group

            MHA_mask_L = MHA_mask.L()
            head_mask_L = head_mask.L()
            FFN_mask_L = FFN_mask.L()

            params.append(4 * 64 * hidden_mask.sum() * head_mask_L.sum() * MHA_mask_L.sum())

            mask = torch.outer(hidden_mask, filter_mask.L())
            mask = mask * FFN_mask_L
            params.append(2 * mask.sum())

        s = torch.stack(params).sum() / M
        return s

    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval"
                 ) -> Dict[str, float]:
        # 使用明确的评估开始横幅
        logger.info("=" * 50)
        logger.info("EVALUATION START")
        logger.info("=" * 50)
        
        # 重置评估日志控制标志
        self.eval_log_once = False
        
        # Save current distill switch state
        orig_distill_switch = self.distill_switch

        # Temporarily enable distillation to compute proper loss values
        self.distill_switch = True

        # Calculate current loss values on a sample batch if we're evaluating
        if eval_dataset is not None and hasattr(self, "t_model") and self.t_model is not None:
            # Get a small sample batch for loss calculation
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            if len(eval_dataloader) > 0:
                try:
                    # Get one batch of data
                    batch = next(iter(eval_dataloader))
                    batch = self._prepare_inputs(batch)
                    
                    # 仅在调试模式下记录批次信息
                    if self.debug_mode:
                        logger.debug(f"Evaluation batch keys: {batch.keys()}")
                    
                    # Forward pass through student model
                    with torch.no_grad():
                        # 明确设置 output_hidden_states=True
                        batch_with_hidden = batch.copy()
                        batch_with_hidden["output_hidden_states"] = True
                        outputs = self.model(**batch_with_hidden)
                        
                        # 仅在调试模式下记录输出信息
                        if self.debug_mode:
                            logger.debug(f"Evaluation model output keys: {outputs.keys()}")

                        # Compute distillation loss components
                        _, l_pred, l_layer = self.compute_distill_loss(
                            unwrap_model(self.model),
                            batch,
                            outputs["logits"],
                            outputs["hidden_states"],
                            return_components=True
                        )

                        # Update loss trackers with current values
                        self.last_l_pred = l_pred.item()
                        self.last_l_layer = l_layer.item()
                        self.last_distill_loss = self.args.distill_lambda * l_pred.item() + (
                                    1.0 - self.args.distill_lambda) * l_layer.item()
                except Exception as e:
                    logger.warning(f"Failed to compute evaluation losses: {str(e)[:100]}...")
                    
                    # 在调试模式下记录完整堆栈跟踪
                    if self.debug_mode:
                        import traceback
                        logger.debug(f"Detailed error: {traceback.format_exc()}")
                    
                    # 设置默认损失值，避免后续使用未初始化的值
                    self.last_l_pred = 0.0
                    self.last_l_layer = 0.0
                    self.last_distill_loss = 0.0

        # Log evaluation parameters only once per evaluation call
        if self.args.local_rank == 0 and not self.eval_log_once:
            self.eval_log_once = True  # 标记已经记录
            
            with torch.no_grad():
                lambda_1 = self.model.bert.reg_lambda_1.item()
                lambda_2 = self.model.bert.reg_lambda_2.item()
                sparsity = self.compute_sparsity().item()
                t_sparsity = self.compute_target_sparsity()
                lagrangian_loss = self.compute_lagrangian_loss().item()

                # 使用明确的标识，便于在日志中查找
                logger.info("-" * 40)
                logger.info("[EVAL PARAMS]")
                logger.info(f"λ1: {lambda_1:.4f}, λ2: {lambda_2:.4f}, Sparsity: {sparsity:.4f}/{t_sparsity:.4f}")
                logger.info(f"Distill losses: L_pred={self.last_l_pred:.4f}, L_layer={self.last_l_layer:.4f}, Combined={self.last_distill_loss:.4f}")
                logger.info(f"Lagrangian loss: {lagrangian_loss:.4f}")
                logger.info("-" * 40)

        # Reset distill switch to its original state for the actual evaluation
        self.distill_switch = False

        # Run standard evaluation
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Restore original distill switch state
        self.distill_switch = orig_distill_switch

        # Add loss components to results
        results['l_pred'] = self.last_l_pred
        results['l_layer'] = self.last_l_layer
        results['sparsity'] = self.compute_sparsity().item()
        
        # 使用明确的评估结束横幅
        logger.info("=" * 50)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results: {', '.join([f'{k}={v:.4f}' for k, v in results.items() if isinstance(v, (int, float))])}")
        logger.info("=" * 50)

        return results
        
    # 添加一个方法来控制日志级别
    def set_log_level(self, level="INFO", debug_mode=False):
        """
        设置日志级别和调试模式
        
        参数:
            level: 日志级别，可选 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            debug_mode: 是否启用调试模式打印详细信息
        """
        logger.remove()  # 移除所有处理程序
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            format="<level>{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}</level>",
            level=level,
            colorize=True
        )
        self.debug_mode = debug_mode
        self.log_interval = 100 if debug_mode else 500  # 调试模式下更频繁地记录
        
        logger.info(f"Log level set to {level}, debug mode: {debug_mode}, log interval: {self.log_interval}")
