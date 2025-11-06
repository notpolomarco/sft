from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import Dataset, IterableDataset
from simple_parsing import ArgumentParser
from src.trainer_callbacks import (
    CustomTensorBoardCallback,
)
from src.dataloaders import VenCord
import yaml


def sft(
    config: SFTConfig,
    train_dataset: Union[Dataset, IterableDataset],
    eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
    resume_from_checkpoint: bool = False,
    callbacks: List[TrainerCallback] = [],
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    peft_config: Optional[LoraConfig] = None,
):
    # Device placement is handled automatically by accelerate/deepspeed
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model, tokenizer, _ = clone_chat_template(
    #     model, tokenizer, "HuggingFaceTB/SmolLM2-135M-Instruct"
    # )

    # ---------------------------
    # Initialize trainer
    # ---------------------------
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    for callback in callbacks:
        trainer.add_callback(callback)

    # ---------------------------
    # Optional: custom evaluation callback
    # ---------------------------
    # cb = LLMSampleCB(trainer, formatted_test, max_new_tokens=256)
    # trainer.add_callback(cb)

    try:
        if resume_from_checkpoint:
            print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print("Received interrupt; saving state and model...")
        trainer.save_state()
        trainer.save_model()
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description="sft")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    model_name = config_dict.pop("model_name", "HuggingFaceTB/SmolLM2-135M-Instruct")
    peft_config_dict = config_dict.pop("peft_config", None)

    peft_config = LoraConfig(**peft_config_dict) if peft_config_dict else None

    config = SFTConfig(**config_dict)

    train_dataset = VenCord()
    eval_dataset = train_dataset.shuffle(seed=42).select(range(10))

    tensorboard_callback = CustomTensorBoardCallback(log_dir=config.logging_dir)

    sft(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume,
        callbacks=[tensorboard_callback],
        model_name=model_name,
        peft_config=peft_config,
    )
