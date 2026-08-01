"""Modal deploy entry for SenseNova-Vision (SenseTime, unified vision UMM).

One app, five slots, all backed by sensenova/SenseNova-Vision-7B-MoT (a
BAGEL-7B-MoT derivative that casts CV tasks as unified multimodal
generation — text generation for symbolic outputs, image generation for
dense pixel-aligned outputs):

- ``image-describe``  free-form visual QA / captioning (understanding mode)
- ``image-gen-text``  instruction + optional image -> text; also covers
                      detection / OCR-style structured-text prompts
- ``image-normal``    per-pixel surface normal map (dense_perception mode)
- ``image-matting``   salient-object binary mask as straight-alpha RGBA PNG
- ``image-pose``      COCO-17 human keypoints, rendered as a skeleton overlay

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.image_describe import ImageDescribeInput, ImageDescribeOutput
from tongflow.models.image_gen_text import ImageGenTextInput, ImageGenTextOutput
from tongflow.models.image_matting import ImageMattingInput, ImageMattingOutput
from tongflow.models.image_normal import ImageNormalInput, ImageNormalOutput
from tongflow.models.image_pose import ImagePoseInput, ImagePoseOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

# Slots this plugin is the default implementation of: the node picker lists
# it first and a newly added node preselects it. Read statically by the
# scanner (never executed), so any SDK version imports this file fine.
TONGFLOW_DEFAULT_SLOTS = ["image-describe", "image-gen-text"]

REPO_URL = "https://github.com/OpenSenseNova/SenseNova-Vision.git"
# Pin the upstream revision so redeploys are reproducible (main moves).
REPO_REV = "beea1f771b8192c597085c0e668f4430d2089d8d"
REPO_DIR = "/app/SenseNova-Vision"

# Prebuilt wheel matching torch 2.5.1 (cxx11abiFALSE) / cu12 / cp310 — the
# bagel modeling code hard-imports flash_attn; building from source is ~30 min.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

_HERE = Path(__file__).resolve().parent

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(_HERE.name)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "fonts-dejavu-core",
        # open3d runtime libs (imported at module load by inference/utils_3d)
        "libegl1", "libgomp1",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        "transformers==4.49.0",
        "accelerate==1.7.0",
        "einops==0.8.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.29.1",
        "opencv-python-headless==4.11.0.86",
        "numpy==1.26.4",
        "Pillow==11.2.1",
        "tqdm==4.67.1",
        # inference/utils_3d imports these at module load even though the
        # five mounted slots never run recon3d / camera pose.
        "open3d==0.19.0",
        "trimesh==4.12.2",
        "scipy==1.15.3",
    )
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout {REPO_REV}",
    )
    .pip_install("tongflow==0.2.20")
    .env(
        {
            "HF_HOME": "/models/hf",
            "SENSENOVA_VISION_WEIGHTS": "/models/sensenova-vision/SenseNova-Vision-7B-MoT",
            "PYTHONPATH": f"{REPO_DIR}:/opt/sensenova_plugin",
        }
    )
    # Mounted at runtime (copy defaults to False) so every deploy ships the
    # latest glue without baking a cacheable image layer.
    .add_local_file(
        str(_HERE / "sensenova_runtime.py"),
        "/opt/sensenova_plugin/sensenova_runtime.py",
    )
)

with image.imports():
    import sensenova_runtime as rt


@deploy
@app.cls(
    image=image,
    # Resolved at deploy time; changing it requires a redeploy.
    gpu=os.environ.get("SENSENOVA_VISION_GPU", "L40S"),
    memory=32768,
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=300,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        self.model = rt.load_model()

    @modal.method()
    @node_slot(NodeSlots.IMAGE_DESCRIBE)
    def image_describe(self, input: ImageDescribeInput) -> ImageDescribeOutput:
        """Free-form image description / visual QA."""
        try:
            instruction = (
                (input.userPrompt or "").strip()
                or (input.text or "").strip()
                or rt.DESCRIBE_FALLBACK
            )
            text = rt.understand(
                self.model, instruction, prompt_media_to_bytes(input.image)
            )
        except Exception as e:
            return ImageDescribeOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageDescribeOutput(success=True, text=text)

    @modal.method()
    @node_slot(NodeSlots.IMAGE_GEN_TEXT)
    def image_gen_text(self, input: ImageGenTextInput) -> ImageGenTextOutput:
        """Instruction (+ optional image) -> text.

        Detection / OCR style prompts return the model's structured text
        (<bbox> / <kpt> tags). There is no system channel in the model's chat
        format, so `system` is prepended to the instruction; top_p/top_k are
        not supported by the upstream sampler and are ignored.
        """
        try:
            question = input.text
            if input.system and input.system.strip():
                question = f"{input.system.strip()}\n\n{question}"
            img = (
                prompt_media_to_bytes(input.image)
                if input.image is not None
                else None
            )
            text = rt.understand(
                self.model,
                question,
                img,
                max_tokens=input.max_new_tokens,
                think=bool(input.enable_thinking),
                temperature=input.temperature,
            )
        except Exception as e:
            return ImageGenTextOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageGenTextOutput(success=True, text=text)

    @modal.method()
    @node_slot(NodeSlots.IMAGE_NORMAL)
    def image_normal(self, input: ImageNormalInput) -> ImageNormalOutput:
        """Per-pixel surface normal map at the input resolution."""
        try:
            data = rt.normal_map_png(self.model, prompt_media_to_bytes(input.image))
        except Exception as e:
            return ImageNormalOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageNormalOutput(
            success=True, image=asset(data, mime="image/png", filename="normal.png")
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_MATTING)
    def image_matting(self, input: ImageMattingInput) -> ImageMattingOutput:
        """Salient-object segmentation as a straight-alpha transparent PNG."""
        try:
            data = rt.matting_png(self.model, prompt_media_to_bytes(input.image))
        except Exception as e:
            return ImageMattingOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImageMattingOutput(
            success=True, image=asset(data, mime="image/png", filename="matting.png")
        )

    @modal.method()
    @node_slot(NodeSlots.IMAGE_POSE)
    def image_pose(self, input: ImagePoseInput) -> ImagePoseOutput:
        """COCO-17 human keypoint overlay for every detected person."""
        try:
            data = rt.pose_png(self.model, prompt_media_to_bytes(input.image))
        except Exception as e:
            return ImagePoseOutput(success=False, error=f"{type(e).__name__}: {e}")
        return ImagePoseOutput(
            success=True, image=asset(data, mime="image/png", filename="pose.png")
        )
