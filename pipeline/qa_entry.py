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
    EvalPrediction,
    DataCollatorWithPadding
)

from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertForQuestionAnswering as TModel

from modeling.modeling_cofi_bert import CoFiBertForQuestionAnswering as SModel

from datasets import DatasetDict, Dataset, load_from_disk, load_metric, load_dataset
from typing import Optional, Dict, List, Tuple, Callable, Union
from copy import deepcopy
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


def setup_seed(training_args):
    seed: int = training_args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_logger(training_args):
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level
    )


def get_distill_args(args):
    distill_args = deepcopy(args)
    distill_args.num_train_epochs = args.distill_num_train_epochs
    distill_args.learning_rate = args.distill_learning_rate
    distill_args.evaluation_strategy = "epoch"

    return distill_args


def get_num_params(model: nn.Module):
    num_params = 0
    num_params_without_residual = 0
    for name, params in model.named_parameters():
        if 'encoder' in name:
            num_params += params.view(-1).shape[0]
            if 'residual' not in name:
                num_params_without_residual += params.view(-1).shape[0]
    return num_params, num_params_without_residual


def prepare_dataset(
        args,
        training_args,
):
    
    raw_datasets = load_from_disk('./datasets/squad')

    tokenizer = AutoTokenizer.from_pretrained("./model")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    utils = BertSquadUtils()
    train_map_fn = utils.get_train_map_fn(training_args, tokenizer)
    validation_map_fn = utils.get_validation_map_fn(training_args, tokenizer)
    
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
    
    metric = load_metric("./pipeline/qa_metric.py")
    
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    post_processing_function = utils.get_post_processing_fn(training_args)
    
    #raw_datasets['train'] = raw_datasets['train'].filter(lambda x: len(x["input_ids"]) <= 512)
    #raw_datasets['validation'] = raw_datasets['validation_matched'].filter(lambda x: len(x["input_ids"]) <= 512)

    return raw_datasets, tokenizer, compute_metrics, post_processing_function


def run():
    args, training_args = parse_hf_args()

    setup_seed(training_args)
    setup_logger(training_args)
    
    datasets, tokenizer, compute_metrics, post_processing_function = prepare_dataset(args, training_args)
    data_collator = DataCollatorWithPadding(tokenizer)
    datasets['train'] = datasets['train'].shard(num_shards=2, index=0, contiguous=True)

    train_dataset = datasets['train']
    eval_dataset = datasets['validation']
    eval_examples = datasets["validation_examples"]

    t_model = TModel.from_pretrained('./model')
    training_args.num_train_epochs = 5
    training_args.gradient_accumulation_steps = 2
    training_args.logging_strategy  = "epoch"
    training_args.evaluation_strategy  = "epoch"
    training_args.save_strategy  = "epoch"

    trainer = DefaultTrainer(
            t_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics = compute_metrics,
        )
    
    train_result = trainer.train()

    