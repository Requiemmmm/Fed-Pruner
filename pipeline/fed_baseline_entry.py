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
    DataCollatorWithPadding,
    Trainer, # Use standard Trainer
    TrainingArguments as HfTrainingArguments, # Use standard TrainingArguments for Trainer
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification, # Use standard AutoModel
    set_seed,
)
# Removed CoFiBert import

from datasets import DatasetDict, Dataset, load_from_disk, load_metric, load_dataset

from typing import Optional, Dict, List, Tuple, Callable, Union
from copy import deepcopy
# Removed DistillTrainer import

# Import the baseline args we defined
from .baseline_args import ModelArguments, TrainingArguments 

log = logging.getLogger(__name__)

# --- Helper function to get GLUE task info ---
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_num_labels(task_name):
    # Simplified: assumes GLUE tasks. Needs extension for other datasets.
    if task_name == "stsb":
        return 1
    elif task_name == "mnli":
        return 3
    else: # cola, mrpc, qnli, qqp, rte, sst2, wnli
        return 2

# --- Client Class ---
class Client():
    # Simplified init: Receives necessary components
    def __init__(self, client_id: int, train_dataset: Dataset, eval_dataset: Dataset, 
                 tokenizer: AutoTokenizer, data_collator: DataCollatorWithPadding, 
                 compute_metrics: Callable, args: TrainingArguments):
        
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset # Global validation set for potential local eval
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.args = args # Baseline TrainingArguments

        # Create standard HuggingFace TrainingArguments for the Trainer
        # Note: We map FL args (rounds, local_epochs) to Trainer's args if needed
        # For simplicity, assume Trainer runs for args.local_epochs
        self.hf_training_args = HfTrainingArguments(
            output_dir=os.path.join(args.output_dir, f"client_{client_id}_tmp"), # Temporary dir for trainer state
            num_train_epochs=args.local_epochs, # Train for local_epochs
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            # --- Add other relevant standard TrainingArguments ---
            logging_dir=os.path.join(args.output_dir, f"client_{client_id}_logs"),
            logging_steps=100, # Log less frequently in local training
            evaluation_strategy="no", # Typically no evaluation during local FL training
            save_strategy="no", # Don't save checkpoints locally
            disable_tqdm=True, # Disable progress bars for cleaner logs
            report_to=[], # Disable reporting to wandb etc.
            seed=args.seed + client_id, # Ensure different seed per client per round if desired
            # --- DP arguments are handled outside the Trainer ---
        )

    def train_epoch(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Train the model locally for one FL round (multiple local epochs).
        Applies DP noise to the *returned weights* if enabled.
        """
        log.info(f"Client {self.client_id}: Starting local training for {self.args.local_epochs} epochs.")
        
        # Ensure model is on the correct device
        model.to(self.hf_training_args.device) 
        
        # Instantiate standard Trainer
        trainer = Trainer(
            model=model,
            args=self.hf_training_args, # Use standard HF TrainingArguments
            train_dataset=self.train_dataset,
            # eval_dataset=self.eval_dataset, # Optional: can evaluate locally if needed
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            # --- No Teacher model, no custom loss logic needed here ---
        )
        
        # Train the model (runs for self.args.local_epochs)
        train_result = trainer.train()
        log.info(f"Client {self.client_id}: Local training finished. Result: {train_result}")

        # Get updated weights
        weights = model.state_dict()
        
        # Apply Differential Privacy noise to weights before returning
        if self.args.apply_dp:
            log.info(f"Client {self.client_id}: Applying DP noise (epsilon={self.args.dp_epsilon}, delta={self.args.dp_delta})")
            # Simplified Laplace noise application (assuming sensitivity C=1 for L2 norm, which needs justification)
            # Note: A more robust DP implementation might use Gaussian noise with accountant, 
            # and noise added *during* training (DP-SGD) via Opacus or similar.
            # This post-hoc Laplace noise addition is a simplification.
            # Sensitivity might depend on clipping norm C and dataset size n (sensitivity = 2*C/n for Laplace over avg gradient)
            # Here we approximate sensitivity based on dataset size - THIS IS LIKELY INCORRECT & NEEDS REVISITING
            # For demonstration, let's assume fixed sensitivity or derive sigma directly.
            # A common approach (though potentially weak privacy) is fixed sigma based on epsilon.
            # We use the formula from the original code for consistency, but highlight its potential issues.
            
            # WARNING: The sensitivity calculation below is likely incorrect for adding noise
            # directly to weights. It was originally used assuming gradient averaging.
            # Robust DP for FL is complex. For baseline, ensure consistency with Fed-Pruner's DP.
            # If Fed-Pruner used this method, use it here for fair comparison, but acknowledge limitations.
            
            # sensitivity = 2 * self.hf_training_args.max_grad_norm / len(self.train_dataset) # More typical sensitivity if adding noise to average gradient
            # A simpler approach is to determine sigma directly based on desired privacy level. Let's follow original code's calculation:
            
            # Make sure train_dataset has len attribute or get its size
            dataset_size = len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else 1000 # Estimate if no len
            
            if dataset_size == 0:
                 log.warning(f"Client {self.client_id}: Train dataset size is 0, skipping DP noise.")
                 return weights

            sensitivity = 2 / dataset_size # Original calculation, assuming C=1? Needs justification.
            sigma = sensitivity / self.args.dp_epsilon # Calculate scale for Laplace noise

            if sigma > 0: # Only add noise if sigma is positive
                for name, param in weights.items():
                    if param.requires_grad: # Add noise only to trainable parameters
                        noise = torch.from_numpy(
                            np.random.laplace(loc=0, scale=sigma, size=param.shape)
                        ).to(param.device, dtype=param.dtype)
                        weights[name] = param + noise
                log.info(f"Client {self.client_id}: Applied Laplace noise with scale sigma={sigma:.4f}")
            else:
                 log.warning(f"Client {self.client_id}: DP sigma <= 0 ({sigma:.4f}), skipping noise application.")

        # Detach weights from graph and move to CPU before returning
        return {k: v.detach().cpu().clone() for k, v in weights.items()}


# --- Server Class ---
class Server():
    # Simplified init: Receives necessary components
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, 
                 client_train_datasets: List[Dataset], validation_dataset: Dataset, 
                 data_collator: DataCollatorWithPadding, compute_metrics: Callable, 
                 args: TrainingArguments):
                 
        self.model = model # The single baseline model
        self.tokenizer = tokenizer
        self.client_train_datasets = client_train_datasets
        self.validation_dataset = validation_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.args = args

        # Create clients
        self.clients = [
            Client(client_id=i, train_dataset=client_data, eval_dataset=validation_dataset, 
                   tokenizer=tokenizer, data_collator=data_collator, 
                   compute_metrics=compute_metrics, args=args) 
            for i, client_data in enumerate(client_train_datasets)
        ]
        
        self.num_clients = len(self.clients)
        log.info(f"Server initialized with {self.num_clients} clients.")
        
        self.best_result = 0.0 # Track best validation accuracy

        # Setup HF Trainer args for server-side evaluation
        self.hf_eval_args = HfTrainingArguments(
            output_dir=os.path.join(args.output_dir, "server_eval_tmp"),
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            logging_dir=os.path.join(args.output_dir, "server_eval_logs"),
            report_to=[],
            seed=args.seed,
            # Other args don't matter much for evaluation only
        )


    def distribute_task(self, client_ids: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Distribute current global model to selected clients and collect updated weights."""
        server_weights_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        client_weight_datas = []
        
        for client_id in client_ids:
            log.info(f"Distributing task to client {client_id}...")
            client = self.clients[client_id]
            # Create a fresh copy of the model for each client's training run
            local_model = deepcopy(self.model) 
            # Pass the model object itself to the client
            
            #updated_weights_cpu = client.train_epoch(local_model, client_id) 
            updated_weights_cpu = client.train_epoch(local_model) 
            client_weight_datas.append(updated_weights_cpu)
            log.info(f"Received updated weights from client {client_id}.")
            del local_model # Free memory
            torch.cuda.empty_cache() # Try to free GPU memory if applicable
            
        return client_weight_datas
        
    def federated_average(self, client_weight_datas: List[Dict[str, torch.Tensor]]):
        """Aggregate client weights using FedAvg."""
        if not client_weight_datas:
            log.warning("Received empty list of client weights. Skipping aggregation.")
            return

        client_num = len(client_weight_datas)
        log.info(f"Aggregating weights from {client_num} clients.")
        
        # Initialize aggregated weights with the first client's weights
        aggregated_weights = deepcopy(client_weight_datas[0]) 
        
        # Sum weights from other clients
        for i in range(1, client_num):
            for key in aggregated_weights:
                aggregated_weights[key] += client_weight_datas[i][key]
                
        # Average the weights
        for key in aggregated_weights:
            aggregated_weights[key] = torch.div(aggregated_weights[key], client_num)
            
        # Load aggregated weights into the server model
        self.model.load_state_dict(aggregated_weights)
        log.info("Federated averaging complete. Updated global model.")

    def evaluate(self, round_num: int):
        """Evaluate the global model on the validation set."""
        log.info(f"--- Evaluating global model after round {round_num} ---")
        
        self.model.to(self.hf_eval_args.device) # Ensure model is on correct device

        # Instantiate standard Trainer for evaluation
        eval_trainer = Trainer(
            model=self.model,
            args=self.hf_eval_args,
            eval_dataset=self.validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        results = eval_trainer.evaluate()
        
        # --- Removed sparsity check ---
        accuracy_key = 'eval_accuracy' # Default key
        # Check if accuracy key exists, handle potential variations (e.g., 'eval_acc')
        if accuracy_key not in results:
             possible_keys = [k for k in results.keys() if 'acc' in k.lower()]
             if possible_keys:
                  accuracy_key = possible_keys[0]
             else:
                  log.error(f"Could not find accuracy key in evaluation results: {results}")
                  return # Cannot proceed without accuracy

        current_accuracy = results[accuracy_key]
        log.info(f"Round {round_num} Evaluation Results: {results}")
        
        if current_accuracy > self.best_result:
            self.best_result = current_accuracy
            log.info(f"*** New best accuracy: {self.best_result:.4f} at round {round_num} ***")
            # Optionally save the best model checkpoint here
            # output_dir = os.path.join(self.args.output_dir, f"checkpoint-best")
            # eval_trainer.save_model(output_dir) 
            # log.info(f"Saved best model checkpoint to {output_dir}")

        log.info(f"Current Best Accuracy: {self.best_result:.4f}")
        print(f"Round {round_num} Evaluation Results: {results}") # Keep console print
        print(f"Current Best Accuracy: {self.best_result:.4f}")   # Keep console print
    
    def run(self):
        """Run the federated learning process."""
        log.info("Starting Federated Learning (Baseline)...")
        # Initial evaluation (Round 0)
        self.evaluate(round_num=0) 
        
        for r in range(self.args.rounds):
            round_num = r + 1
            log.info(f"===== Starting Round {round_num}/{self.args.rounds} =====")
            
            # Client selection (simple: all clients) - can be extended
            selected_client_ids = list(range(self.num_clients))
            log.info(f"Selected clients for round {round_num}: {selected_client_ids}")
            
            # Distribute task, train locally, collect weights
            client_weights = self.distribute_task(selected_client_ids)
            
            # Aggregate weights
            self.federated_average(client_weights)
            
            # Evaluate global model
            self.evaluate(round_num=round_num)
            
            # --- Removed target sparsity update ---
            
        log.info("Federated Learning (Baseline) finished.")
        log.info(f"Final Best Accuracy: {self.best_result:.4f}")




        # --- Main Function ---
# 修改函数签名以接收 model_args 和 training_args
# --- Main Function ---
# 修改函数签名以接收 model_args 和 training_args
def main(model_args: ModelArguments, training_args: TrainingArguments): 

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    log.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 使用正确的参数对象打印日志
    log.info(f"Baseline Model Arguments: {model_args}") 
    log.info(f"Baseline Training Arguments: {training_args}")
    
    # Set seed before initializing model/data
    # 使用 training_args
    set_seed(training_args.seed)

    # --- Load Dataset ---
    # 使用 training_args
    log.info(f"Loading dataset: GLUE task '{training_args.dataset_name}'") 
    # 构建本地数据集路径
    # 使用 training_args
    local_dataset_path = os.path.join("./datasets", training_args.dataset_name) 
    
    log.info(f"Attempting to load dataset '{training_args.dataset_name}' from local path: '{local_dataset_path}'") # 使用 training_args
    
    # *** 修改点：强制检查并从本地加载，失败则退出 ***
    if os.path.exists(local_dataset_path):
        try:
            # 从本地磁盘加载
            raw_datasets = load_from_disk(local_dataset_path) 
            log.info(f"Successfully loaded dataset from '{local_dataset_path}'.")
        except Exception as e:
            # 如果本地加载失败，则报错退出
            log.error(f"Failed to load dataset from local path '{local_dataset_path}'. Ensure data is intact. Error: {e}")
            sys.exit(1)
    else:
        # 如果本地路径不存在，则报错退出
        log.error(f"Local dataset path '{local_dataset_path}' not found. Please ensure the dataset exists in the './datasets' directory.")
        sys.exit(1)

    # --- 后续代码 ---
    
    # Determine number of labels
    # 使用 training_args
    is_regression = training_args.dataset_name == "stsb"
    if not is_regression:
        # Ensure 'train' split exists before accessing features
        if "train" not in raw_datasets:
            log.error(f"'train' split not found in loaded dataset: {raw_datasets.keys()}")
            sys.exit(1)
        # Check if 'label' feature exists and has 'names'
        if "label" not in raw_datasets["train"].features or not hasattr(raw_datasets["train"].features["label"], 'names'):
             log.error(f"'label' feature with 'names' attribute not found in 'train' split features: {raw_datasets['train'].features}")
             num_labels = get_num_labels(training_args.dataset_name) # Fallback to helper if needed # 使用 training_args
             log.warning(f"Using fallback num_labels: {num_labels}")
        else:
             label_list = raw_datasets["train"].features["label"].names
             num_labels = len(label_list)
        log.info(f"Task is classification. Number of labels: {num_labels}")
    else:
        num_labels = 1
        log.info("Task is regression (STSB).")
        
    # --- Load Tokenizer and Config ---
    # 使用 model_args
    log.info(f"Loading tokenizer and config for model: '{model_args.model_name_or_path}'") 
    try:
         config = AutoConfig.from_pretrained(
             model_args.model_name_or_path, # <--- 使用 model_args
             num_labels=num_labels, 
             finetuning_task=training_args.dataset_name # <--- 使用 training_args (任务名)
        )
         # ***修改点：使用 model_args 加载 tokenizer***
         tokenizer = AutoTokenizer.from_pretrained(
             model_args.model_name_or_path, # <--- 使用 model_args
             use_fast=model_args.use_fast  # <--- 使用 model_args
        )
         # 检查并添加 [PAD] token (如果需要)
         if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            log.info("Added [PAD] token to tokenizer.")

    except Exception as e:
         log.error(f"Failed to load tokenizer/config for '{model_args.model_name_or_path}'. Ensure path is correct and model files exist. Error: {e}") # 使用 model_args
         sys.exit(1)
         
    # --- Load Model ---
    # 使用 model_args
    log.info(f"Loading pre-trained model: '{model_args.model_name_or_path}'")
    try:
         model = AutoModelForSequenceClassification.from_pretrained(
             model_args.model_name_or_path, # <--- 使用 model_args
             config=config,
         )
         # 如果 tokenizer 被修改了，调整模型嵌入大小
         # 应该在添加 special token 之后，加载模型权重 *之前* 或 *之后* 都可以调整
         # 如果在加载权重之后调整，只会初始化新增 token 的 embedding
         model.resize_token_embeddings(len(tokenizer))
         log.info(f"Resized model embeddings to fit tokenizer size: {len(tokenizer)}")
    except Exception as e:
         log.error(f"Failed to load model '{model_args.model_name_or_path}'. Ensure path is correct and model files exist. Error: {e}") # 使用 model_args
         sys.exit(1)

    # --- Preprocess Dataset ---
    # 使用 training_args
    sentence1_key, sentence2_key = task_to_keys[training_args.dataset_name]
    # 使用 training_args
    padding = "max_length" if training_args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        # 使用 training_args
        result = tokenizer(*texts, padding=padding, max_length=training_args.max_seq_length, truncation=True)

        if "label" in examples:
            result["label"] = examples["label"]
        return result

    log.info("Preprocessing datasets...")
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True, # Enable caching
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    # 使用 training_args
    validation_key = "validation_matched" if training_args.dataset_name == "mnli" else "validation"
    # ... (rest of eval_dataset selection logic remains the same) ...
    if validation_key not in processed_datasets:
         log.warning(f"Validation key '{validation_key}' not found. Using 'validation'.")
         validation_key = "validation"
         if validation_key not in processed_datasets:
              log.error(f"Neither 'validation_matched' nor 'validation' found in dataset splits: {processed_datasets.keys()}")
              log.warning("Using train split for validation as fallback.")
              eval_dataset = train_dataset 
         else:
             eval_dataset = processed_datasets[validation_key]
    else:
         eval_dataset = processed_datasets[validation_key]


    # --- Split training data for clients ---
    # 使用 training_args
    log.info(f"Splitting training data for {training_args.num_clients} clients...")
    client_train_datasets = []
    # 使用 training_args
    num_shards_divisor = training_args.num_clients * 2 if training_args.half else training_args.num_clients
    # 使用 training_args
    start_index = training_args.num_clients if training_args.half else 0

    # ... (rest of client data sharding logic remains the same) ...
    if len(train_dataset) < num_shards_divisor:
         log.warning(f"Training dataset size ({len(train_dataset)}) is smaller than the number of shards ({num_shards_divisor}). Distributing data might be uneven or fail.")
         for i in range(training_args.num_clients): # 使用 training_args
             client_train_datasets.append(train_dataset)
    else:
         try:
             for i in range(training_args.num_clients): # 使用 training_args
                  shard_index = start_index + i
                  client_train_datasets.append(
                      train_dataset.shard(num_shards=num_shards_divisor, index=shard_index, contiguous=True)
                  )
                  log.info(f"Client {i} assigned shard {shard_index}/{num_shards_divisor}, size: {len(client_train_datasets[-1])}")
         except IndexError as e:
              log.error(f"Error sharding dataset: {e}. Dataset size: {len(train_dataset)}, num_shards: {num_shards_divisor}. Check 'half' argument and client number.")
              sys.exit(1)

             
    # --- Setup Data Collator ---
    # 使用 training_args (假设 fp16 在 training_args 中)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if hasattr(training_args, 'fp16') and training_args.fp16 else None)

    # --- Setup Compute Metrics ---
    # 使用 training_args
    metric = load_metric("glue", training_args.dataset_name)
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # 使用 is_regression
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        main_metric = list(result.keys())[0] 
        return {main_metric: result[main_metric]}


    # --- Create Server and Run ---
    log.info("Initializing Server...")
    # 传递 training_args 给 Server/Client
    server = Server(
        model=model,
        tokenizer=tokenizer,
        client_train_datasets=client_train_datasets,
        validation_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        args=training_args, 
    )
    
    server.run()

# Note: This script now expects to be called with arguments defined in baseline_args.py
# Example: python glue_fedavg_baseline_train.py --dataset_name sst2 --num_clients 5 --rounds 10 ...