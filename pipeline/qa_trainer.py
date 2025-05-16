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

from transformers import Trainer 
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

import math
import time

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput, speed_metrics

SModel = CoFiBertForSequenceClassification
class DefaultTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        start_time = time.time()
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        if self.post_process_function is not None and self.compute_metrics is not None and self.args.should_save:
            # Only the main node write the results by default
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            metrics.update(output.metrics)
        else:
            metrics = output.metrics

        if self.args.should_log:
            # Only the main node log the results by default
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        start_time = time.time()
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        metrics.update(output.metrics)
        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
    

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
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 eval_examples=None,
                 post_process_function=None,
                 ):
        assert callbacks is None
        super().__init__(
            s_model, args, data_collator, train_dataset, eval_dataset, 
            tokenizer, model_init, compute_metrics, callbacks, optimizers, 
            preprocess_logits_for_metrics,
            eval_examples=eval_examples, 
            post_process_function=post_process_function,
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

        self.reg_params = []

    
        self.per_layer_mask_groups: List[Tuple[Mask, ...]] = []
        self.init_reg_params()
        self.ffn_masks: List[Mask] = []
        self.init_ffn_masks()
    
        
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
        result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self.distill_switch = False


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
            distill_loss = self.compute_distill_loss(
                unwrap_model(model),
                inputs,
                outputs["start_logits"], 
                outputs["end_logits"], 
                outputs["hidden_states"]
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

    def compute_distill_loss(self, 
        model: SModel,
        inputs: Dict,
        s_start_logits: torch.Tensor,
        s_end_logits: torch.Tensor,
        s_hidden_states: torch.Tensor,
    ):
        with torch.no_grad():
            assert "output_hidden_states" in inputs and inputs["output_hidden_states"] is True
            t_outputs = self.t_model(**inputs)
            t_start_logits = t_outputs["start_logits"]
            t_end_logits = t_outputs["end_logits"]
            t_hidden_states = t_outputs["hidden_states"]

        mask: torch.Tensor = inputs["attention_mask"]
        D = s_start_logits.shape[-1]
        T = self.args.distill_T
        distill_lambda = self.args.distill_lambda
        
        pred_loss = self.kl_loss(
            torch.log_softmax(s_start_logits / T, dim=-1),
            torch.log_softmax(t_start_logits / T, dim=-1),
        ) + self.kl_loss(
            torch.log_softmax(s_end_logits / T, dim=-1),
            torch.log_softmax(t_end_logits / T, dim=-1),
        )
        
        assert len(t_hidden_states) == len(s_hidden_states)
        
        proj = model.bert.distill_projection
        t_hidden_states = [self.mask_select(t_h, mask) for t_h in t_hidden_states]
        s_hidden_states = [proj(self.mask_select(s_h, mask)) for s_h in s_hidden_states]
        
        match_index = []
        with torch.no_grad():
            T = torch.stack(t_hidden_states).unsqueeze(0)
            S = torch.stack(s_hidden_states).unsqueeze(1)
            dist = (T - S).pow(2.).mean(-1).mean(-1)  # dist[i, j] = || S_i - T_j ||
            assert len(dist.shape) == 2
        
        num_layers = len(s_hidden_states)
        for i in range(num_layers):
            match_index.append(dist[i, i:].argmin().item() + i)
        
        _layer_loss = []
        for i, s_h in enumerate(s_hidden_states):
            t_h = t_hidden_states[match_index[i]]
            _layer_loss.append(self.mse_loss(t_h, s_h))
        layer_loss = torch.stack(_layer_loss).sum()
        
        distill_loss = distill_lambda * pred_loss + (1.0 - distill_lambda) * layer_loss
        
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
        ffn_size = 768*4
        M = (hidden_size * hidden_size * 4 + hidden_size * ffn_size * 2) * num_layers
        params = []
        hidden_mask = torch.ones([768]).cuda()
        for mask_group in self.per_layer_mask_groups:
            head_mask, MHA_mask, FFN_mask, filter_mask = mask_group

            
            MHA_mask_L = MHA_mask.L()
            head_mask_L = head_mask.L()
            FFN_mask_L = FFN_mask.L()

            params.append(4  * 64 * hidden_mask.sum() * head_mask_L.sum() * MHA_mask_L.sum())

            mask = torch.outer(hidden_mask, filter_mask.L())
            mask = mask * FFN_mask_L
            params.append(2 * mask.sum())

        s = torch.stack(params).sum() / M
        return s

    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 eval_examples: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval"
                 ) -> Dict[str, float]:
        if self.args.local_rank == 0:
            with torch.no_grad():
                lambda_1 = self.model.bert.reg_lambda_1.item()
                lambda_2 = self.model.bert.reg_lambda_2.item()
                sparsity = self.compute_sparsity()
                t_sparsity = self.compute_target_sparsity()
                lagrangian_loss = self.compute_lagrangian_loss()
                logger.info("lambda-1: {}".format(lambda_1))
                logger.info("lambda-2: {}".format(lambda_2))
                logger.info("sparsity = {}".format(sparsity))
                logger.info("t_sparsity = {}".format(t_sparsity))
                logger.info("lagrangian_loss = {}".format(lagrangian_loss))
                #time = str(datetime.datetime.now())
                #sparsity_ = np.array(sparsity.cpu())
                #np.save(time, sparsity_)

        past_distill_switch = self.distill_switch
        self.distill_switch = False
        results = super().evaluate(eval_dataset, eval_examples, ignore_keys, metric_key_prefix)
        self.distill_switch = past_distill_switch
        
        with torch.no_grad():
            results['sparsity'] = self.compute_sparsity()
            
        #print("best_result: ", self.best_result)
 
        return results