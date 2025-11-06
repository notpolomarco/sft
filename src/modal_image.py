import modal
from pathlib import Path


MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app_name = "sft"
app = modal.App(app_name)

volume = modal.Volume.from_name("sft-volume", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.13",
    )
    .uv_pip_install(
        "datasets>=4.3.0",
        "deepspeed>=0.18.1",
        "hf-transfer>=0.1.9",
        "modal>=1.2.1",
        "torch>=2.9.0",
        "trl>=0.24.0",
        "tensorboard>=2.20.0",
        "peft>=0.17.1",
        "wandb>=0.22.3",
        "simple-parsing>=0.1.7",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

image = image.add_local_dir(Path(__file__).parent, remote_path="/root/src")
image = image.add_local_dir(
    Path(__file__).parent.parent / "data" / "vencord", remote_path="/root/data/vencord"
)
image = image.add_local_dir(
    Path(__file__).parent.parent / "config", remote_path="/root/config"
)
