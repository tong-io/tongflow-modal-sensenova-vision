"""Glue around the vendored SenseNova-Vision repo (on PYTHONPATH).

The repo's ``SenseNovaVisionModel`` casts every CV task as unified multimodal
generation: text-output tasks run in ``understanding`` / ``dense_detection``
modes, dense pixel-aligned tasks (normal maps, binary masks) run in
``dense_perception`` mode and return a PIL image.

Task prompts below are copied verbatim from the upstream demo
(``inference/inference_demo.py``) — they are part of the model's training
distribution and must not be paraphrased.
"""

from __future__ import annotations

import io
import os

from PIL import Image

DESCRIBE_FALLBACK = "Describe this image in detail."

NORMAL_PROMPT = (
    "Estimate surface normals and encode as an RGB image. Each channel "
    "corresponds to a direction component (X, Y, Z) with continuous value "
    "variations, creating smooth color gradients distinct from other task outputs."
)

# Referring binary segmentation (MASK_QUESTION_LIST[0] upstream). The query
# phrase for the matting slot is a plugin choice, not an ABI field.
BINARY_SEG_PROMPT_TEMPLATE = (
    "Can you segment the image based on the following categories: "
    "<p>{category}</p>? Please output the binary segmentation masks."
)
MATTING_QUERY = "the most salient foreground object"

# Human keypoint detection (KEYPOINT_PROMPT_TEMPLATE["human"] upstream),
# fixed to <p>person</p> — the image-pose contract has no text input.
KEYPOINT_PROMPT = (
    "Detect all instances of <p>person</p> in the image. For each instance, "
    "output a bounding box in <bbox> format and the coordinates of its "
    "nose, left eye, right eye, left ear, right ear, left shoulder, right "
    "shoulder, left elbow, right elbow, left wrist, right wrist, left hip, "
    "right hip, left knee, right knee, left ankle, right ankle in "
    "<kpt>[x,y]</kpt> format. Return results as a structured list."
)


def load_model():
    from inference.sensenova_vision import SenseNovaVisionModel

    return SenseNovaVisionModel(
        model_path=os.environ["SENSENOVA_VISION_WEIGHTS"],
        dtype="bf16",
        device="cuda",
        max_mem_per_gpu=os.environ.get("SENSENOVA_VISION_MAX_MEM", "44GiB"),
        offload_folder="/tmp/offload",
    )


def _pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def understand(
    model,
    question: str,
    image: bytes | None = None,
    *,
    max_tokens: int | None = None,
    think: bool = False,
    temperature: float | None = None,
    mode: str = "understanding",
) -> str:
    """Text-output inference; returns the generated string."""
    if think and mode == "understanding":
        mode = "think_understanding"
    q = f"<image> {question}" if image is not None else question
    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_think_token_n"] = int(max_tokens)
    if temperature is not None and temperature > 0:
        kwargs["do_sample"] = True
        kwargs["text_temperature"] = float(temperature)
    out = model.generate(
        question=q,
        images=[io.BytesIO(image)] if image is not None else [],
        mode=mode,
        **kwargs,
    )
    if isinstance(out, dict):  # mixed-output warning path
        out = out.get("text") or ""
    return str(out).strip()


def _dense(model, prompt: str, image: bytes) -> Image.Image:
    """dense_perception inference; returns the predicted PIL image."""
    out = model.generate(
        question=f"<image> {prompt}",
        images=[io.BytesIO(image)],
        mode="dense_perception",
    )
    if isinstance(out, dict):
        out = out.get("image")
    if not isinstance(out, Image.Image):
        raise RuntimeError("model returned no image for dense_perception")
    return out


def normal_map_png(model, image: bytes) -> bytes:
    """Surface normal map, resized back to the input resolution."""
    source = _pil(image)
    pred = _dense(model, NORMAL_PROMPT, image)
    if pred.size != source.size:
        pred = pred.resize(source.size, Image.Resampling.BICUBIC)
    return _png(pred.convert("RGB"))


def matting_png(model, image: bytes) -> bytes:
    """Salient-object mask applied as straight alpha over the input."""
    source = _pil(image)
    prompt = BINARY_SEG_PROMPT_TEMPLATE.format(category=MATTING_QUERY)
    mask = _dense(model, prompt, image).convert("L")
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.Resampling.BILINEAR)
    rgba = source.copy()
    rgba.putalpha(mask)
    return _png(rgba)


def pose_png(model, image: bytes) -> bytes:
    """Human keypoints as text, rendered back onto the input image."""
    from utils.visualize import VisualizationConfig, visualize_detection

    source = _pil(image)
    text = understand(model, KEYPOINT_PROMPT, image, mode="dense_detection")
    vis = visualize_detection(
        source,
        text,
        task_name="keypoint",
        prompt=KEYPOINT_PROMPT,
        config=VisualizationConfig(),
        include_prompt=False,
    )
    return _png(vis.convert("RGB"))
