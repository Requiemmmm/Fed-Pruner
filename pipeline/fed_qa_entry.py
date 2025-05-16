import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import gc
import torch.nn as nn
import logging
import transformers
from transformers import (
    HfArgumentParser,
    EvalPrediction,
    DataCollatorWithPadding
)
from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertForQuestionAnswering as TModel

from modeling.modeling_cofi_bert import CoFiBertForQuestionAnswering as SModel

from datasets import DatasetDict, Dataset, load_from_disk, load_metric, load_dataset
from typing import Optional, Dict, List, Tuple, Callable, Union
from copy import deepcopy
from transformers import Trainer
import logging

from .qa_trainer import DefaultTrainer, DistillTrainer
from .qa_utils2 import (
    BertSquadUtils
)
from .qa_args import (
    TrainingArguments,
    ModelArguments,
)
from copy import deepcopy

Trainers = Union[DefaultTrainer, DistillTrainer]

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



class Client():
    def __init__(self, epsilon = 1000, num_clients = 2):
        
        args, training_args = parse_hf_args()

        
        raw_datasets = load_from_disk('./datasets/squad')

        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        utils = BertSquadUtils()
        train_map_fn = utils.get_train_map_fn(training_args, self.tokenizer)
        validation_map_fn = utils.get_validation_map_fn(training_args, self.tokenizer)
    
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            raw_datasets["train"] = raw_datasets["train"].map(
                train_map_fn,
                batched=True,
                remove_columns=utils.column_names,
                desc="Running tokenizer on train dataset",
            )
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            raw_datasets["validation_examples"] = raw_datasets["validation"]
            raw_datasets["validation"] = raw_datasets["validation"].map(
                validation_map_fn,
                batched=True,
                remove_columns=utils.column_names,
                desc="Running tokenizer on validation dataset",
            )
        self.metric = load_metric("./pipeline/qa_metric.py")
        self.post_processing_function = utils.get_post_processing_fn(training_args)
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.epsilon = epsilon
        self.num_clients = num_clients
        self.dataset = raw_datasets
        self.half = training_args.half
        self.client_train_datas = self.load_client_train_datas()
        
        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 1
        self.distill_args.gradient_accumulation_steps = 8
        
    def load_client_train_datas(self):
        client_train_datas = []
        if self.half == False:
            for i in range(self.num_clients):
                client_train_datas.append(self.dataset['train'].shard(num_shards=self.num_clients, index=i, contiguous=True))
        else:
            for i in range(self.num_clients):
                #client_train_datas.append(self.dataset['train'].shard(num_shards=4, index=i, contiguous=True))
                client_train_datas.append(self.dataset['train'].shard(num_shards=self.num_clients*2, index=self.num_clients + i, contiguous=True))
        return client_train_datas

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)
        
    def train_epoch(self, server_model, client_id, server_weights, t_model):
        datasets = self.client_train_datas[client_id]
        server_model.load_state_dict(server_weights)
        
        distill_trainer = DistillTrainer(
            server_model,
            t_model,
            args=self.distill_args,
            train_dataset=datasets,
            eval_dataset=self.dataset['validation'],
            eval_examples=self.dataset['validation_examples'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )
        
        distill_trainer.train()
        weight = server_model.state_dict()
        
        sensitivity = 2 / len(datasets)
        sigma = sensitivity / self.epsilon
        #print(weight)
        for i in weight:
            if i == "bert.reg_lambda_1" or i == "bert.reg_lambda_2" :
                continue
            noise = np.random.laplace(0, sigma, weight[i].shape)
            weight[i] = torch.tensor(noise).cuda() + weight[i]
        
        return weight
    
    

class Server():
    def __init__(self, epochs = 100, num_clients = 2):
        args, training_args = parse_hf_args()
        self.num_clients = num_clients
        self.client = Client()
        self.epochs = epochs
        self.distill = training_args.distill
        if self.distill == True:
            self.t_model = TModel.from_pretrained('./[glue]/squad-half-datas')
            self.s_model = SModel.from_pretrained('./[glue]/squad-half-datas')
        if self.distill == False:
            self.t_model = TModel.from_pretrained('./model')
            self.s_model = SModel.from_pretrained('./model')
        
        
        raw_datasets = load_from_disk('./datasets/squad')

        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        utils = BertSquadUtils()
        train_map_fn = utils.get_train_map_fn(training_args, self.tokenizer)
        validation_map_fn = utils.get_validation_map_fn(training_args, self.tokenizer)
    
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            raw_datasets["train"] = raw_datasets["train"].map(
                train_map_fn,
                batched=True,
                remove_columns=utils.column_names,
                desc="Running tokenizer on train dataset",
            )
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            raw_datasets["validation_examples"] = raw_datasets["validation"]
            raw_datasets["validation"] = raw_datasets["validation"].map(
                validation_map_fn,
                batched=True,
                remove_columns=utils.column_names,
                desc="Running tokenizer on validation dataset",
            )
        self.metric = load_metric("./pipeline/qa_metric.py")
        self.post_processing_function = utils.get_post_processing_fn(training_args)
        
        self.dataset = raw_datasets
        
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
        
    def federated_average(self, client_weight_datas):
        client_num = len(client_weight_datas)
        assert client_num != 0
        w = client_weight_datas[0]
        for i in range(1, client_num):
            for j in w:
                w[j] = w[j] + client_weight_datas[i][j]
        for i in w:
            w[i] = w[i] / client_num
        self.s_model.load_state_dict(w)
        
        del client_weight_datas
        gc.collect()
        return w
    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)
       
    
    def evalute(self):
        Trainer = DistillTrainer(
            self.s_model,
            self.t_model,
            args=self.distill_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )
        results = Trainer.evaluate(eval_dataset=self.dataset['validation'], eval_examples=self.dataset['validation_examples'])
        print(results)
        if results['eval_f1'] > self.best_result and results['sparsity'] < 0.21:
            self.best_result = results['eval_f1']
        print("best_results:", self.best_result)
    
    def run(self):
        for epoch in range(self.epochs):
            print("epoch: ", epoch)
            client_ids = [i for i in range(self.num_clients)]
            #client_ids=[0]
            client_weight_datas = self.distribute_task(client_ids)
            self.federated_average(client_weight_datas)
            self.evalute()
            self.client.distill_args.target_sparsity = max(0.2, self.client.distill_args.target_sparsity - 0.2)
            

        
