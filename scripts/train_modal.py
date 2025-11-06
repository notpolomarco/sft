from pathlib import PosixPath
import modal
from dotenv import load_dotenv
from src.modal_image import app, image, volume

load_dotenv()

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

volume_path = PosixPath("/vol/data")
tb_log_path = volume_path / "tb_logs"
model_save_path = volume_path / "models"
gpu = os.getenv("GPU", "A10G:2")  # Multi-GPU: 2 GPUs by default


with image.imports():
    import yaml
    import os


@app.function(
    image=image,
    volumes={volume_path: volume},
    gpu=gpu,
    timeout=1 * HOURS,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_modal(config_path: str = "config/config.yaml"):
    import subprocess
    import torch
    import tempfile

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs for training")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict["output_dir"] = str(model_save_path)
    config_dict["logging_dir"] = str(tb_log_path)
    config_dict["report_to"] = "wandb"  # Use wandb for Modal

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_dict, tmp)
        tmp_config_path = tmp.name

    try:
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            "config/accelerate_config.yaml",
            "--num_processes",
            str(num_gpus),
            "-m",
            "src.sft",
            "--config",
            tmp_config_path,
        ]

        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(tmp_config_path)


@app.local_entrypoint()
def main(config_path: str = "config/config.yaml"):
    train_modal.remote(config_path=config_path)
