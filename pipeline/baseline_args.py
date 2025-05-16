'''’import os
from dataclasses import dataclass, field
from transformers import TrainingArguments as DefaultTrainingArguments
from transformers.training_args import (
    IntervalStrategy
)
from typing import Optional, Union

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default="bert"
    )
    use_fast: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    
@dataclass
class TrainingArguments(DefaultTrainingArguments):
    
    dataset_name: Optional[str] = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
        default="glue",
    )
    
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    

    target_sparsity: Optional[float] = field(default=0.8)
    # sparsity = (new params number) / (origin params number)
    
    distill_T: float = field(default=2.0)
    distill_lambda: float = field(default=0.3)  # lambda * loss_pred + (1 - lambda) * loss_layer
    
    reg_learning_rate: float = field(default=1e-1)
    
    distill_num_train_epochs: float = field(default=40, metadata={"help": "Total number of training epochs to perform."})
    distill_learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})
    
    
    # Overwrite 
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})

    output_dir: Optional[str] = field(
        metadata={"help": "The name of the task to train on."},
        default=None,
    )
    
    distill: bool = field(default = True)   # 联邦学习过程是否使用蒸馏
    
    half: bool = field(default = True)      # 联邦学习过程是否只使用一半数据
    
    save_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    
    def get_file_name(self):
        return "[{}]".format(
            self.dataset_name,
        )
    
    def __post_init__(self):
        # update output dir
        self.output_dir = self.get_file_name()
    super().__post_init__()'''


import os
from dataclasses import dataclass, field
from transformers import TrainingArguments as DefaultTrainingArguments
from transformers.training_args import (
    IntervalStrategy
)
from typing import Optional, Union

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. Should be 'bert-base-uncased' or similar for baseline."},
       ## default="bert-base-uncased" # Default to standard BERT
        default="./model" 
    )
    use_fast: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    
@dataclass
class TrainingArguments(DefaultTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    
    dataset_name: Optional[str] = field(
        metadata={"help": "The name of the dataset to use (via the datasets library). e.g., 'sst2', 'qqp'"},
        default=None, # Make it task-specific via command line
    )
    
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    
    # === Arguments Removed ===
    # target_sparsity
    # distill_T
    # distill_lambda
    # reg_learning_rate
    # distill_num_train_epochs
    # distill_learning_rate
    # distill
    # =======================
    
    # === Standard Training Arguments (Keep/Adjust Defaults) ===
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."} # Adjusted default potentially
    )
    per_device_eval_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."} # Adjusted default potentially
    )
    
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW for the baseline."}) 

    output_dir: Optional[str] = field(
        default="./output", # Provide a sensible default base output directory
        metadata={"help": "The base output directory where the model predictions and checkpoints will be written."},
    )

    # Ensure you have these or similar arguments defined and KEPT:
    num_clients: int = field(default=5, metadata={"help": "Number of clients in the federated setting."}) # Example - KEEP
    rounds: int = field(default=10, metadata={"help": "Number of communication rounds in federated learning."}) # Example - KEEP
    local_epochs: int = field(default=3, metadata={"help": "Number of local training epochs on each client per round."}) # Example - KEEP
    
    apply_dp: bool = field(default=False, metadata={"help": "Apply differential privacy."}) # Example - KEEP
    dp_epsilon: float = field(default=8.0, metadata={"help": "DP epsilon value."}) # Example - KEEP
    dp_delta: float = field(default=1e-5, metadata={"help": "DP delta value. Should be less than 1/num_samples."}) # Example - KEEP
    # dp_max_grad_norm: float = field(default=1.0, metadata={"help": "DP max grad norm for clipping."}) # Example - KEEP if used

    seed: int = field(default=42, metadata={"help": "Random seed for initialization."}) # Example - KEEP
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."}) # Example - KEEP
    
    half: bool = field(default=False, metadata={"help": "Whether to use only half the training data per client."}) # KEEP THIS if used in Fed-Pruner runs

    save_strategy: Union[IntervalStrategy, str] = field(
        default="no", # Typically 'no' for FL simulations unless saving final model
        metadata={"help": "The checkpoint save strategy to use."},
    )
    
    # You might want a specific flag for the baseline run
    run_name: Optional[str] = field(default="fedavg_baseline", metadata={"help": "A name for this specific run, used in output dir naming."})
    
    def get_output_dir_name(self):
        # Generate a more descriptive output directory name
        return os.path.join(
            self.output_dir,
            f"{self.dataset_name}",
            f"{self.run_name}_clients{self.num_clients}_rounds{self.rounds}_lepochs{self.local_epochs}_lr{self.learning_rate}_seed{self.seed}"
            f"{'_dp_eps' + str(self.dp_epsilon) if self.apply_dp else ''}"
            f"{'_halfdata' if self.half else ''}"
        )
    
    def __post_init__(self):
        # update output dir to be more descriptive
        if self.dataset_name is None:
             raise ValueError("Dataset name must be specified.")
        self.output_dir = self.get_output_dir_name()
        
        # Call the parent's __post_init__ if necessary, depends on DefaultTrainingArguments
        # Check if DefaultTrainingArguments has a __post_init__ you rely on. If unsure, keep it.
        try:
            super().__post_init__()
        except AttributeError: # In case the parent class doesn't have __post_init__
            pass 
        # Add any other necessary post-initialization checks here
        if self.apply_dp and self.dp_delta is None:
             raise ValueError("dp_delta must be specified when apply_dp is True.")

'''主要改动总结：

删除了 target_sparsity, distill_T, distill_lambda, reg_learning_rate, distill_num_train_epochs, distill_learning_rate, distill 这些参数。
保留了 model_name, dataset_name, max_seq_length, pad_to_max_length, per_device_train_batch_size, per_device_eval_batch_size, learning_rate, output_dir, half, save_strategy。
添加/强调了 需要保留的联邦学习参数 (num_clients, rounds, local_epochs) 和差分隐私参数 (apply_dp, dp_epsilon, dp_delta) 以及其他标准参数 (seed, weight_decay) 的占位符/示例。请确保这些参数在你的最终 baseline_args.py 中确实存在且被保留！
修改了 model_name 的默认值为 'bert-base-uncased'，因为基线应该使用标准模型。
修改了 dataset_name 的默认值为 None，强制通过命令行指定任务。
改进了 __post_init__ 中的 output_dir 生成逻辑，使其包含更多超参数信息，便于区分不同的实验运行。你可以根据需要调整这个命名逻辑。'''
        
