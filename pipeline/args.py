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

    # ========== 量化相关参数 ==========
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
        metadata={
            "help": "Client quantization strategy: 'local_only' (quantize locally, send float), 'send_quantized' (send quantized weights), 'adaptive' (choose based on network conditions)."}
    )

    # ========== 差分隐私相关参数 ==========
    apply_dp: bool = field(
        default=False,
        metadata={"help": "Whether to apply differential privacy to federated learning."}
    )

    dp_target_epsilon: float = field(
        default=10.0,
        metadata={
            "help": "Target privacy budget (epsilon) for the entire training process. Lower values provide stronger privacy. Typical values: 1.0-10.0 for strong privacy, 10.0-100.0 for moderate privacy."}
    )

    dp_target_delta: float = field(
        default=1e-5,
        metadata={
            "help": "Target privacy parameter (delta). Should be much smaller than 1/dataset_size. Typical value: 1e-5."}
    )

    dp_noise_multiplier: float = field(
        default=1.0,
        metadata={
            "help": "Noise multiplier for differential privacy. Higher values add more noise for stronger privacy. Will be auto-computed if dp_auto_compute_noise is True."}
    )

    dp_clipping_bound: float = field(
        default=1.0,
        metadata={
            "help": "Clipping bound for client gradients/updates. Updates with norm larger than this will be clipped. Affects both privacy and utility."}
    )

    dp_accountant_type: str = field(
        default="rdp",
        metadata={
            "help": "Privacy accountant type: 'rdp' (Rényi Differential Privacy, recommended), 'gdp' (Gaussian Differential Privacy), or 'ma' (Moments Accountant)."}
    )

    # DP算法优化参数
    dp_auto_clip: bool = field(
        default=True,
        metadata={
            "help": "Whether to automatically adjust clipping bound based on gradient norms. Helps optimize utility."}
    )

    dp_adaptive_noise: bool = field(
        default=True,
        metadata={
            "help": "Whether to adaptively adjust noise based on remaining privacy budget. Can improve utility in later rounds."}
    )

    dp_auto_compute_noise: bool = field(
        default=True,
        metadata={
            "help": "Whether to automatically compute noise multiplier from target epsilon/delta. If False, uses dp_noise_multiplier directly."}
    )

    # DP安全和调试参数
    dp_secure_mode: bool = field(
        default=True,
        metadata={"help": "Whether to enable secure mode with additional privacy safeguards and validation."}
    )

    dp_compute_batch_size: int = field(
        default=64,
        metadata={
            "help": "Effective batch size for DP computations. Should match or be close to actual training batch size."}
    )

    # 服务器端DP参数
    server_learning_rate: float = field(
        default=1.0,
        metadata={
            "help": "Server-side learning rate for applying aggregated updates. Used in DP-FedAvg. Typical values: 0.1-1.0."}
    )

    # DP调试和分析参数
    dp_verbose_logging: bool = field(
        default=False,
        metadata={"help": "Whether to enable verbose DP logging for debugging and analysis."}
    )

    dp_save_privacy_history: bool = field(
        default=True,
        metadata={"help": "Whether to save privacy budget consumption history to file."}
    )

    # DP高级参数
    dp_sampling_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Client sampling rate per round. If None, computed as num_selected_clients / total_clients."}
    )

    dp_max_grad_norm: Optional[float] = field(
        default=None,
        metadata={
            "help": "Maximum gradient norm before clipping. If None, uses dp_clipping_bound. This is an alias for backward compatibility."}
    )

    def get_file_name(self):
        """生成包含量化和DP信息的文件名"""
        suffix_parts = []

        # 量化后缀
        if self.apply_quantization:
            suffix_parts.append(f"quant_{self.quantization_type}")

        # DP后缀
        if self.apply_dp:
            # 使用较短的DP标识，避免文件名过长
            dp_suffix = f"dp_eps{self.dp_target_epsilon}"
            if self.dp_noise_multiplier != 1.0:
                dp_suffix += f"_nm{self.dp_noise_multiplier}"
            suffix_parts.append(dp_suffix)

        # 组合所有后缀
        if suffix_parts:
            suffix = "_" + "_".join(suffix_parts)
        else:
            suffix = ""

        return "[{}]{}".format(self.dataset_name, suffix)

    def _validate_quantization_args(self):
        """验证量化参数"""
        if not self.apply_quantization:
            return

        if self.quantization_type not in ["dynamic", "static"]:
            raise ValueError(f"Invalid quantization_type: {self.quantization_type}. Must be 'dynamic' or 'static'.")

        if self.quantization_backend not in ["fbgemm", "qnnpack"]:
            raise ValueError(
                f"Invalid quantization_backend: {self.quantization_backend}. Must be 'fbgemm' or 'qnnpack'.")

        if self.client_quantization_strategy not in ["local_only", "send_quantized", "adaptive"]:
            raise ValueError(f"Invalid client_quantization_strategy: {self.client_quantization_strategy}")

        if self.calibration_batch_size <= 0:
            raise ValueError("calibration_batch_size must be positive")

        if self.num_calibration_batches <= 0:
            raise ValueError("num_calibration_batches must be positive")

    def _validate_dp_args(self):
        """验证差分隐私参数"""
        if not self.apply_dp:
            return

        # 基本范围检查
        if self.dp_target_epsilon <= 0:
            raise ValueError("dp_target_epsilon must be positive")

        if self.dp_target_delta <= 0 or self.dp_target_delta >= 1:
            raise ValueError("dp_target_delta must be in (0, 1)")

        if self.dp_noise_multiplier < 0:
            raise ValueError("dp_noise_multiplier must be non-negative")

        if self.dp_clipping_bound <= 0:
            raise ValueError("dp_clipping_bound must be positive")

        # 隐私会计类型检查
        valid_accountants = ["rdp", "gdp", "ma"]
        if self.dp_accountant_type not in valid_accountants:
            raise ValueError(f"dp_accountant_type must be one of {valid_accountants}")

        # 服务器学习率检查
        if self.server_learning_rate <= 0:
            raise ValueError("server_learning_rate must be positive")

        # 批大小检查
        if self.dp_compute_batch_size <= 0:
            raise ValueError("dp_compute_batch_size must be positive")

        # 采样率检查（如果提供）
        if self.dp_sampling_rate is not None:
            if self.dp_sampling_rate <= 0 or self.dp_sampling_rate > 1:
                raise ValueError("dp_sampling_rate must be in (0, 1]")

        # 最大梯度范数检查（向后兼容性）
        if self.dp_max_grad_norm is not None:
            if self.dp_max_grad_norm <= 0:
                raise ValueError("dp_max_grad_norm must be positive")
            # 如果同时设置了两个参数，发出警告
            if self.dp_max_grad_norm != self.dp_clipping_bound:
                import warnings
                warnings.warn(
                    f"Both dp_max_grad_norm ({self.dp_max_grad_norm}) and dp_clipping_bound ({self.dp_clipping_bound}) "
                    "are set with different values. Using dp_clipping_bound."
                )

    def _validate_compatibility(self):
        """验证不同功能之间的兼容性"""
        # DP和传统噪声的兼容性检查
        if self.apply_dp and self.global_noise_type != "none" and self.global_noise_scale > 0:
            import warnings
            warnings.warn(
                "Both differential privacy (apply_dp=True) and legacy global noise "
                "(global_noise_type != 'none') are enabled. This may lead to over-noising. "
                "Consider using only differential privacy for formal privacy guarantees."
            )

        # DP和量化的兼容性（应该是兼容的，但给出信息）
        if self.apply_dp and self.apply_quantization:
            import warnings
            warnings.warn(
                "Both differential privacy and quantization are enabled. "
                "This is supported but may require careful tuning of DP parameters."
            )

    def get_dp_summary(self):
        """获取DP配置摘要（用于日志输出）"""
        if not self.apply_dp:
            return "DP: Disabled"

        summary = f"DP: ε={self.dp_target_epsilon}, δ={self.dp_target_delta}, "
        summary += f"σ={self.dp_noise_multiplier}, C={self.dp_clipping_bound}, "
        summary += f"accountant={self.dp_accountant_type}"

        if self.dp_auto_clip:
            summary += ", auto-clip"
        if self.dp_adaptive_noise:
            summary += ", adaptive-noise"
        if self.dp_auto_compute_noise:
            summary += ", auto-noise"

        return summary

    def __post_init__(self):
        # 向后兼容性：如果设置了dp_max_grad_norm但没设置dp_clipping_bound
        if hasattr(self, 'dp_max_grad_norm') and self.dp_max_grad_norm is not None:
            if not hasattr(self, 'dp_clipping_bound') or self.dp_clipping_bound == 1.0:
                self.dp_clipping_bound = self.dp_max_grad_norm

        # 验证所有参数
        self._validate_quantization_args()
        self._validate_dp_args()
        self._validate_compatibility()

        # 更新输出目录
        self.output_dir = self.get_file_name()

        # 调用父类的post_init
        super().__post_init__()
