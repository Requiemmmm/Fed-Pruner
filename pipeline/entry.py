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
    dataset = load_from_disk('./datasets/sst2')  # 换数据集这里要改成对应的数据集

    tokenizer = AutoTokenizer.from_pretrained("./model")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(example):
        # return tokenizer(example["premise"], example["hypothesis"], truncation=True)  #换数据集时这里需要改   qqp：question1 question2   qnli：question  sentence  mnli：premise hypothesis
        return tokenizer(example["sentence"], truncation=True)  # sst2数据集用这行

    dataset = dataset.map(tokenize_function, batched=True)
    dataset['train'] = dataset['train'].shard(num_shards=2, index=0, contiguous=True)  # 这行表示只用一半数据训练，我们只需要使用一半数据训练的模型
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return {"accuracy": accuracy}

    dataset['train'] = dataset['train'].filter(lambda x: len(x["input_ids"]) <= 512)
    # dataset['validation'] = dataset['validation_matched'].filter(lambda x: len(x["input_ids"]) <= 512)  #mnli数据集用这行
    dataset['validation'] = dataset['validation'].filter(lambda x: len(x["input_ids"]) <= 512)  # 其余三个数据集用这行

    return dataset, tokenizer, compute_metrics, data_collator


def run():
    args, training_args = parse_hf_args()

    setup_seed(training_args)
    setup_logger(training_args)
    datasets, tokenizer, compute_metrics, data_collator = prepare_dataset(args, training_args)

    train_dataset = datasets['train']
    eval_dataset = datasets['validation']

    # t_model = TModel.from_pretrained('./model')
    # --- 修改开始 ---
    # 从数据集中获取标签数量 (对于 MNLI 应该是 3)
    num_labels = datasets['train'].features['label'].num_classes
    print(f"****** Setting num_labels to: {num_labels} ******")  # 添加打印方便确认

    # 在加载模型时传入 num_labels
    t_model = TModel.from_pretrained('./model', num_labels=num_labels)
    # --- 修改结束 ---

    training_args.num_train_epochs = 5
    training_args.gradient_accumulation_steps = 2
    training_args.logging_strategy = "epoch"
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "epoch"

    trainer = Trainer(
        t_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    # 训练结束后，查看哪一个epoch结果最好，保留对应的checkpoint文件夹，改名为数据集名-half-datas，如mnli-half-datas
