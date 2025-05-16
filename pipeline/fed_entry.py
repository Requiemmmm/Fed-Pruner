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
from collections import OrderedDict  # 确保导入 OrderedDict

from .trainer import DistillTrainer
from .args import (
    TrainingArguments,
    ModelArguments,
)
from copy import deepcopy

Trainers = Union[Trainer, DistillTrainer]


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

    print(f"INFO: Applying server-side Laplacian noise with scale b={noise_scale_b}")
    noisy_state_dict = OrderedDict()  # Create a new dict for noisy parameters
    for key, param_tensor in state_dict.items():
        if param_tensor.is_floating_point():  # <<< ONLY ADD NOISE TO FLOATING POINT TENSORS
            current_param_device = param_tensor.device
            current_param_dtype = param_tensor.dtype

            noise_values = np.random.laplace(loc=0.0, scale=noise_scale_b, size=param_tensor.shape)
            noise_tensor = torch.from_numpy(noise_values).to(device=current_param_device, dtype=current_param_dtype)

            noisy_state_dict[key] = param_tensor + noise_tensor
        else:
            noisy_state_dict[key] = param_tensor.clone()  # Copy non-float tensors as is
    return noisy_state_dict


class Client():
    def __init__(self, epsilon=1000, num_clients=2):

        args, training_args = parse_hf_args()
        dataset = load_from_disk('./datasets/sst2')  # 换数据集需要改
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(example):
            # return self.tokenizer(example["premise"], example["hypothesis"], truncation=True)  #换数据集时这里需要改   qqp：question1 question2   qnli：question  sentence  mnli：premise hypothesis
            return self.tokenizer(example["sentence"], truncation=True)  # sst2数据集用这行

        dataset = dataset.map(tokenize_function, batched=True)
        dataset['train'] = dataset['train'].filter(lambda x: len(x["input_ids"]) <= 512)
        # dataset['validation'] = dataset['validation_matched'].filter(lambda x: len(x["input_ids"]) <= 512)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.epsilon = epsilon
        self.num_clients = num_clients
        self.dataset = dataset
        self.half = training_args.half
        self.client_train_datas = self.load_client_train_datas()

        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 1
        self.distill_args.gradient_accumulation_steps = 4

    def load_client_train_datas(self):
        client_train_datas = []
        if self.half == False:
            for i in range(self.num_clients):
                client_train_datas.append(
                    self.dataset['train'].shard(num_shards=self.num_clients, index=i, contiguous=True))
        else:
            for i in range(self.num_clients):
                # client_train_datas.append(self.dataset['train'].shard(num_shards=4, index=i, contiguous=True))
                client_train_datas.append(
                    self.dataset['train'].shard(num_shards=self.num_clients * 2, index=self.num_clients + i,
                                                contiguous=True))
        return client_train_datas

    def compute_metrics(self, eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return {"accuracy": accuracy}

    def train_epoch(self, server_model, client_id, server_weights, t_model):
        datasets = self.client_train_datas[client_id]
        server_model.load_state_dict(server_weights)

        distill_trainer = DistillTrainer(
            server_model,
            t_model,
            args=self.distill_args,
            train_dataset=datasets,
            eval_dataset=self.dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        distill_trainer.train()
        weight = server_model.state_dict()
        '''
        sensitivity = 2 / len(datasets)
        sigma = sensitivity / self.epsilon
        for i in weight:
            if i == "bert.reg_lambda_1" or i == "bert.reg_lambda_2" :
                continue
            noise = np.random.laplace(0, sigma, weight[i].shape)
            weight[i] = torch.tensor(noise).cuda() + weight[i]'''

        return weight


class Server():
    def __init__(self, epochs=100, num_clients=2):
        args, training_args = parse_hf_args()
        self.training_args = training_args
        self.num_clients = num_clients
        self.client = Client()
        self.epochs = epochs
        self.distill = training_args.distill

        if self.distill == True:
            self.t_model = TModel.from_pretrained('./[glue]/sst2-half-datas')  # 换数据集时这里需要改
            self.s_model = SModel.from_pretrained('./[glue]/sst2-half-datas')  # 换数据集时这里需要改
        if self.distill == False:
            self.t_model = TModel.from_pretrained('./model')
            self.s_model = SModel.from_pretrained('./model')

        dataset = load_from_disk('./datasets/sst2')  # 换数据集时这里需要改
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(example):
            # return self.tokenizer(example["premise"], example["hypothesis"], truncation=True)    #换数据集时这里需要改   qqp：question1 question2   qnli：question  sentence  mnli：premise hypothesis
            return self.tokenizer(example["sentence"], truncation=True)  # sst2数据集用这行

        dataset = dataset.map(tokenize_function, batched=True)
        # dataset['validation'] = dataset['validation_matched'].filter(lambda x: len(x["input_ids"]) <= 512)   #mnli用这行
        dataset['validation'] = dataset['validation'].filter(lambda x: len(x["input_ids"]) <= 512)  # 其他三个数据集用这行

        self.dataset = dataset['validation']

        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 1

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.best_result = 0

    def distribute_task(self, client_ids):
        server_weights = deepcopy(self.s_model.state_dict())
        client_weight_datas = []

        for i in range(len(client_ids)):
            client_id = client_ids[i]
            weight = deepcopy(self.client.train_epoch(self.s_model, client_id, server_weights, self.t_model))
            client_weight_datas.append(weight)

        return client_weight_datas

    '''def federated_average(self, client_weight_datas):
        client_num = len(client_weight_datas)
        assert client_num != 0
        w = client_weight_datas[0]
        for i in range(1, client_num):
            for j in w:
                w[j] = w[j] + client_weight_datas[i][j]
        for i in w:
            w[i] = w[i] / client_num
        self.s_model.load_state_dict(w)
        return w'''

    # In Fed-Pruner/pipeline/fed_entry.py
    # Within the Server class:

    def federated_average(self, client_weight_datas):
        client_num = len(client_weight_datas)
        assert client_num != 0

        first_client_w = client_weight_datas[0]
        aggregated_w = OrderedDict()  # Use a new OrderedDict to store aggregated results

        print("INFO: Starting federated aggregation...")
        for key in first_client_w.keys():
            param_template = first_client_w[key]  # Get a template parameter for dtype, device, shape

            if param_template.is_floating_point():
                # Aggregate floating-point tensors (weights, biases, etc.)
                sum_param = torch.zeros_like(param_template)
                for i in range(client_num):
                    # Ensure client tensor is on the same device as sum_param before adding
                    sum_param += client_weight_datas[i][key].to(sum_param.device)

                aggregated_w[key] = sum_param / client_num
                # print(f"DEBUG: Averaged float tensor: {key}")
            else:
                # For non-floating-point tensors (e.g., boolean masks, integer counters)
                # Default FedAvg behavior: take the value from the first client.
                # Alternative: check if all clients have the same value, then use it.
                # For now, we take from the first client as a simple strategy.
                aggregated_w[key] = param_template.clone()
                print(f"INFO: Non-float tensor '{key}' (dtype: {param_template.dtype}) "
                      f"not averaged numerically. Value taken from first client.")

        # 2. Add server-side post-processing noise (logic from your previous step)
        final_w = aggregated_w

        if hasattr(self.training_args, 'global_noise_type') and \
                self.training_args.global_noise_type.lower() == 'laplace' and \
                hasattr(self.training_args, 'global_noise_scale') and \
                self.training_args.global_noise_scale > 0:
            final_w = add_laplacian_noise_to_state_dict(aggregated_w, self.training_args.global_noise_scale)

        # 3. Load final weights into the server model
        # Using strict=False can be a temporary workaround if there are unexpected keys
        # (e.g. if client models evolve to have different sets of non-float params),
        # but it's better to aim for strict=True by correctly handling all keys.
        try:
            self.s_model.load_state_dict(final_w, strict=True)
        except RuntimeError as e:
            print(f"ERROR: Failed to load state_dict with strict=True: {e}")
            print("INFO: Attempting to load state_dict with strict=False...")
            self.s_model.load_state_dict(final_w, strict=False)
            print("INFO: Loaded state_dict with strict=False. Please review model structure and aggregated keys.")

        print("INFO: Federated aggregation and model update complete.")
        return final_w

    def compute_metrics(self, eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return {"accuracy": accuracy}

    def evalute(self):
        distill_trainer = DistillTrainer(
            self.s_model,
            self.t_model,
            args=self.distill_args,
            eval_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        results = distill_trainer.evaluate(eval_dataset=self.dataset)
        if results['eval_accuracy'] > self.best_result and results['sparsity'] < 0.11:
            self.best_result = results['eval_accuracy']
        print(results)
        print("best_results:", self.best_result)

    def run(self):
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            client_ids = [i for i in range(self.num_clients)]
            client_weight_datas = self.distribute_task(client_ids)
            self.federated_average(client_weight_datas)
            self.evalute()
            self.client.distill_args.target_sparsity = max(0.1, self.client.distill_args.target_sparsity - 0.2)




