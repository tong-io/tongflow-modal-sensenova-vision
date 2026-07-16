# tongflow-modal-sensenova-vision

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Unified computer vision with **SenseNova-Vision** (`sensenova/SenseNova-Vision-7B-MoT`, a BAGEL-7B-MoT derivative by SenseTime), running on a GPU via [Modal](https://modal.com). One checkpoint casts heterogeneous CV tasks as native text / image generation and serves all five slots.

## Capabilities

- **Image description** (`image-describe`) — free-form captioning and visual QA.
- **Image → text** (`image-gen-text`) — instruction-driven visual understanding; detection / OCR-style prompts return structured text with `<bbox>` / `<kpt>` tags.
- **Surface normals** (`image-normal`) — per-pixel RGB-encoded normal map (dense generation).
- **Matting** (`image-matting`) — salient-object binary mask applied as a straight-alpha transparent PNG.
- **Pose** (`image-pose`) — COCO-17 human keypoints for every detected person, rendered as a skeleton overlay.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin downloads the weights (~30 GB) to a Modal volume and deploys to your Modal account automatically, caching the build. The `sensenova/SenseNova-Vision-7B-MoT` weights are public — no Hugging Face token required.

## Notes

- Runs on an **L40S** by default (`SENSENOVA_VISION_GPU` env at deploy time overrides). The 7B checkpoint loads in bf16 (~15 GB VRAM).
- Dense tasks (normals, matting) are diffusion-decoded (50 timesteps) and take noticeably longer than text tasks.
- The upstream repo is pinned by revision in `deploy.py`; task prompts are copied verbatim from the upstream demo — do not paraphrase them, they are part of the training distribution.
- Upstream: [OpenSenseNova/SenseNova-Vision](https://github.com/OpenSenseNova/SenseNova-Vision) (Apache 2.0).
