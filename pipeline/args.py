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

    global_noise_type: Optional[str] = field(
        default="none",
        metadata={
            "help": "Type of noise to add to global model parameters post-aggregation (options: none, laplace). Default: none."}
    )
    global_noise_scale: float = field(
        default=0.0,
        metadata={
            "help": "Scale of the global noise (e.g., b for Laplace). Only used if global_noise_type is not 'none'. Default: 0.0."}
    )

    target_sparsity: Optional[float] = field(default=0.8)

    distill_T: float = field(default=2.0)
    distill_lambda: float = field(default=0.3)

    reg_learning_rate: float = field(default=1e-1)

    distill_num_train_epochs: float = field(default=40,
                                            metadata={"help": "Total number of training epochs to perform."})
    distill_learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})

    # Overwrite
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})

    output_dir: Optional[str] = field(
        metadata={"help": "The name of the task to train on."},
        default=None,
    )

    distill: bool = field(default=True)
    half: bool = field(default=True)

    save_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The checkpoint save strategy to use."},
    )

    # === 新增量化相关参数 ===
    apply_quantization: bool = field(
        default=False, 
        metadata={"help": "Whether to apply post-training quantization to client models."}
    )
    
    quantization_type: str = field(
        default="dynamic", 
        metadata={"help": "Type of quantization ('dynamic' or 'static')."}
    )
    
    quantization_backend: str = field(
        default="fbgemm", 
        metadata={"help": "Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)."}
    )
    
    # 静态量化相关参数
    calibration_batch_size: int = field(
        default=8, 
        metadata={"help": "Batch size for static quantization calibration."}
    )
    
    num_calibration_batches: int = field(
        default=10, 
        metadata={"help": "Number of batches for static quantization calibration."}
    )
    
    # 服务器端量化选项
    quantize_global_model: bool = field(
        default=False,
        metadata={"help": "Whether to quantize the global model on server before distribution."}
    )
    
    # 通信相关
    simulate_communication_cost: bool = field(
        default=True,
        metadata={"help": "Whether to simulate and log communication cost reduction from quantization."}
    )
    
    # 客户端量化策略
    client_quantization_strategy: str = field(
        default="local_only",
        metadata={"help": "Client quantization strategy: 'local_only' (quantize locally, send float), 'send_quantized' (send quantized weights), 'adaptive' (choose based on network conditions)."}
    )

    def get_file_name(self):
        quant_suffix = ""
        if self.apply_quantization:
            quant_suffix = f"_quant_{self.quantization_type}"
        
        return "[{}]{}".format(
            self.dataset_name,
            quant_suffix
        )

    def __post_init__(self):
        # 验证量化参数
        if self.apply_quantization:
            if self.quantization_type not in ["dynamic", "static"]:
                raise ValueError(f"Invalid quantization_type: {self.quantization_type}")
            
            if self.quantization_backend not in ["fbgemm", "qnnpack"]:
                raise ValueError(f"Invalid quantization_backend: {self.quantization_backend}")
            
            if self.client_quantization_strategy not in ["local_only", "send_quantized", "adaptive"]:
                raise ValueError(f"Invalid client_quantization_strategy: {self.client_quantization_strategy}")
        
        # update output dir
        self.output_dir = self.get_file_name()
        super().__post_init__()
