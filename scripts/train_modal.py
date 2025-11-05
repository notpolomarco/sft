from pathlib import PosixPath
import modal
from src.modal_image import app, image, volume

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

volume_path = PosixPath("/vol/data")
tb_log_path = volume_path / "tb_logs"
model_save_path = volume_path / "models"
gpu = "A10G"


with image.imports():
    from src.sft import sft
    from src.dataloaders import VenCord
    from trl import SFTConfig


@app.function(
    image=image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=1 * HOURS,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_modal():
    config = SFTConfig(
        output_dir=str(model_save_path),
        logging_dir=str(tb_log_path),
        max_steps=1000,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_steps=100,
        eval_strategy="epoch",
        report_to="wandb",
        num_train_epochs=1,
    )

    train_dataset = VenCord()
    eval_dataset = train_dataset.shuffle(seed=42).select(range(10))

    sft(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=False,
        callbacks=[],
    )


@app.local_entrypoint()
def main():
    train_modal.remote()
