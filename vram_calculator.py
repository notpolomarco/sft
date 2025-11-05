import torch
from transformers import AutoModelForCausalLM
import argparse


def get_model_info(model):
    """
    Extract parameter counts and dtype information from model.

    Args:
        model: HuggingFace model

    Returns:
        dict: Model information including param counts, dtype, bytes per param
    """
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtype = next(model.parameters()).dtype
    bytes_per_param = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    return {
        "num_params": num_params,
        "trainable_params": trainable_params,
        "dtype": dtype,
        "bytes_per_param": bytes_per_param,
    }


def calculate_model_weights_memory(num_params, bytes_per_param):
    """
    Calculate memory required for model weights.

    Args:
        num_params: Total number of parameters
        bytes_per_param: Bytes per parameter (2 for fp16/bf16, 4 for fp32)

    Returns:
        int: Memory in bytes
    """
    return num_params * bytes_per_param


def calculate_gradients_memory(trainable_params, bytes_per_param):
    """
    Calculate memory required for gradients.

    Args:
        trainable_params: Number of trainable parameters
        bytes_per_param: Bytes per parameter

    Returns:
        int: Memory in bytes
    """
    return trainable_params * bytes_per_param


def calculate_optimizer_memory(trainable_params, bytes_per_param, optimizer="adamw"):
    """
    Calculate memory required for optimizer states.

    Args:
        trainable_params: Number of trainable parameters
        bytes_per_param: Bytes per parameter (for model)
        optimizer: Optimizer type ('adamw', '8bit_adamw', 'sgd')

    Returns:
        tuple: (total_memory_bytes, breakdown_dict)
    """
    optimizer = optimizer.lower()
    breakdown = {}

    if optimizer == "adamw":
        # AdamW: stores m, v (FP32 each) + FP32 master weights if mixed precision
        m_bytes = 4
        v_bytes = 4
        master_bytes = 4 if bytes_per_param < 4 else 0
        optimizer_bytes_per_param = master_bytes + m_bytes + v_bytes

        breakdown["master_copy_gb"] = (master_bytes * trainable_params) / 1e9
        breakdown["momentum_gb"] = (m_bytes * trainable_params) / 1e9
        breakdown["variance_gb"] = (v_bytes * trainable_params) / 1e9

        optimizer_memory = trainable_params * optimizer_bytes_per_param

    elif optimizer == "8bit_adamw":
        # 8-bit AdamW: quantized m, v (1 byte each) + FP32 master copy
        optimizer_bytes_per_param = 4 + 1 + 1  # 6 bytes total

        breakdown["master_copy_gb"] = (trainable_params * 4) / 1e9
        breakdown["momentum_8bit_gb"] = (trainable_params * 1) / 1e9
        breakdown["variance_8bit_gb"] = (trainable_params * 1) / 1e9

        optimizer_memory = trainable_params * optimizer_bytes_per_param

    elif optimizer == "sgd":
        # SGD with momentum: 1 momentum buffer (FP32)
        if bytes_per_param < 4:
            # Mixed precision — requires FP32 master weights
            optimizer_bytes_per_param = 4 + 4  # master + momentum
        else:
            optimizer_bytes_per_param = 4  # momentum only

        optimizer_memory = trainable_params * optimizer_bytes_per_param
        breakdown["sgd_momentum_gb"] = optimizer_memory / 1e9

    else:
        optimizer_memory = 0
        breakdown["unknown"] = True

    return optimizer_memory, breakdown


def calculate_activations_memory(
    model, batch_size, seq_length, bytes_per_activation, gradient_checkpointing=False
):
    """
    Calculate memory required for activations during training.

    Args:
        model: HuggingFace model (to extract config)
        batch_size: Batch size per device
        seq_length: Sequence length
        bytes_per_activation: Bytes per activation value
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        tuple: (total_memory_bytes, config_dict)
    """
    config = model.config
    hidden_size = getattr(config, "hidden_size", 512)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", 12))

    # Constants per EleutherAI approximations
    k_linear = 4
    k_attn = 1

    linear_activations = (
        batch_size
        * seq_length
        * hidden_size
        * num_layers
        * k_linear
        * bytes_per_activation
    )

    attention_activations = (
        batch_size
        * num_heads
        * (seq_length**2)
        * num_layers
        * k_attn
        * bytes_per_activation
    )

    activation_memory = linear_activations + attention_activations

    if gradient_checkpointing:
        activation_memory *= 0.25  # ~75% reduction

    config_info = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "gradient_checkpointing": gradient_checkpointing,
    }

    return activation_memory, config_info


def calculate_training_memory(
    model, batch_size=4, seq_length=512, optimizer="adamw", gradient_checkpointing=False
):
    """
    Calculate estimated VRAM requirements for LLM training.

    Args:
        model: HuggingFace model
        batch_size: Batch size per device
        seq_length: Max sequence length
        optimizer: Optimizer type ('adamw', '8bit_adamw', 'sgd')
        gradient_checkpointing: Enable gradient checkpointing (reduces activations)

    Returns:
        dict: Memory breakdown (in bytes)
    """

    # Get model information
    model_info = get_model_info(model)
    num_params = model_info["num_params"]
    trainable_params = model_info["trainable_params"]
    dtype = model_info["dtype"]
    bytes_per_param = model_info["bytes_per_param"]
    bytes_per_activation = bytes_per_param  # activations stored in same dtype

    # Print model information
    print(f"\n{'='*70}")
    print(f"MODEL INFORMATION")
    print(f"{'='*70}")
    print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"Model dtype: {dtype}")
    print(f"Bytes per parameter: {bytes_per_param}")

    print(f"\n{'='*70}")
    print(f"MEMORY BREAKDOWN")
    print(f"{'='*70}")

    # 1. Model Weights
    model_memory = calculate_model_weights_memory(num_params, bytes_per_param)
    print(f"\n1. Model Weights: {model_memory / 1e9:.3f} GB")

    # 2. Gradients
    gradient_memory = calculate_gradients_memory(trainable_params, bytes_per_param)
    print(f"\n2. Gradients: {gradient_memory / 1e9:.3f} GB")

    # 3. Optimizer States
    print(f"\n3. Optimizer States ({optimizer}):")
    optimizer_memory, optimizer_breakdown = calculate_optimizer_memory(
        trainable_params, bytes_per_param, optimizer
    )

    # Print optimizer breakdown
    if "master_copy_gb" in optimizer_breakdown:
        print(f"   - FP32 master copy: {optimizer_breakdown['master_copy_gb']:.3f} GB")
    if "momentum_gb" in optimizer_breakdown:
        print(f"   - Momentum (m): {optimizer_breakdown['momentum_gb']:.3f} GB")
    if "variance_gb" in optimizer_breakdown:
        print(f"   - Variance (v): {optimizer_breakdown['variance_gb']:.3f} GB")
    if "momentum_8bit_gb" in optimizer_breakdown:
        print(
            f"   - Quantized momentum (8-bit): {optimizer_breakdown['momentum_8bit_gb']:.3f} GB"
        )
    if "variance_8bit_gb" in optimizer_breakdown:
        print(
            f"   - Quantized variance (8-bit): {optimizer_breakdown['variance_8bit_gb']:.3f} GB"
        )
    if "sgd_momentum_gb" in optimizer_breakdown:
        pass  # Total will be printed below
    if optimizer_breakdown.get("unknown"):
        print("   - Unknown optimizer, assuming 0 GB")

    print(f"   - Total: {optimizer_memory / 1e9:.3f} GB")

    # 4. Activations
    print(f"\n4. Activations (estimated):")
    activation_memory, config_info = calculate_activations_memory(
        model, batch_size, seq_length, bytes_per_activation, gradient_checkpointing
    )

    print(f"   - Hidden size: {config_info['hidden_size']}")
    print(f"   - Layers: {config_info['num_layers']}")
    print(f"   - Attention heads: {config_info['num_heads']}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_length}")

    if config_info["gradient_checkpointing"]:
        print(f"   - Gradient checkpointing: ENABLED (75% reduction)")
    else:
        print(f"   - Gradient checkpointing: DISABLED")

    print(f"   - Estimated total: {activation_memory / 1e9:.3f} GB")

    # Total memory
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory

    print(f"\n{'='*70}")
    print(f"TOTAL ESTIMATED VRAM: {total_memory / 1e9:.3f} GB")
    print(f"{'='*70}\n")

    return {
        "model_gb": model_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "total_gb": total_memory / 1e9,
        "num_params": num_params,
        "trainable_params": trainable_params,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate VRAM requirements for LLM training"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "8bit_adamw", "sgd"]
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Computation dtype",
    )

    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")

    # Map string dtype → torch dtype object
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = None  # "auto" mode lets HF infer dtype

    # Try loading model (using `dtype`, fallback to old `torch_dtype` if needed)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
    except TypeError:
        # Older Transformers versions still use `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
        )

    # Calculate memory usage
    memory_breakdown = calculate_training_memory(
        model=model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        optimizer=args.optimizer,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    print(f"\nSummary:")
    print(f"  Model: {args.model}")
    print(f"  Parameters: {memory_breakdown['num_params']/1e9:.2f}B")
    print(f"  Required VRAM: {memory_breakdown['total_gb']:.2f} GB")


if __name__ == "__main__":
    main()
