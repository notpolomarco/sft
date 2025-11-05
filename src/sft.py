from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer
from trl.models.utils import clone_chat_template
import torch
from peft import LoraConfig
from datasets import Dataset, IterableDataset
from simple_parsing import ArgumentParser
from src.trainer_callbacks import (
    CustomTensorBoardCallback,
)
from src.dataloaders import VenCord


def sft(
    config: SFTConfig,
    train_dataset: Union[Dataset, IterableDataset],
    eval_dataset: Union[Dataset, IterableDataset],
    resume_from_checkpoint: bool = False,
    callbacks: List[TrainerCallback] = [],
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model, tokenizer, _ = clone_chat_template(
    #     model, tokenizer, "HuggingFaceTB/SmolLM2-135M-Instruct"
    # )

    peft_config = LoraConfig(r=256, lora_alpha=16, target_modules="all-linear")

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
    parser.add_arguments(SFTConfig, dest="config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    args = parser.parse_args()

    train_dataset = VenCord()
    eval_dataset = train_dataset.shuffle(seed=42).select(range(10))

    tensorboard_callback = CustomTensorBoardCallback(log_dir=args.config.logging_dir)

    sft(
        config=args.config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume,
        callbacks=[tensorboard_callback],
    )
