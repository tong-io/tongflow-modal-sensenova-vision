"""Modal download entry for SenseNova-Vision.

Run:
  modal run download.py::download

Fetches sensenova/SenseNova-Vision-7B-MoT (~30 GB: ema.safetensors checkpoint,
VAE, tokenizer, configs) into the shared ``models`` volume.

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import modal

MODEL_REPO = "sensenova/SenseNova-Vision-7B-MoT"
MODEL_DIR = "/models/sensenova-vision/SenseNova-Vision-7B-MoT"

volume = modal.Volume.from_name("models", create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==1.6.0")
    .env({"HF_HOME": "/models/hf"}),
    volumes={"/models": volume},
    timeout=7200,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
    print(f"Cached {MODEL_REPO} -> {MODEL_DIR}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
