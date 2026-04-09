# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# One-shot setup (creates venv, installs all deps, runs env check)
bash install.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install "git+https://github.com/openai/shap-e.git"   # shap-e is NOT on PyPI
```

## Running

```bash
source venv/bin/activate

# Validate environment before first run
python check_env.py

# Launch local Gradio app (opens browser at http://localhost:7860)
python app.py
```

## Architecture

**Pipeline flow** (end-to-end):
```
User text prompt
  → pipeline/enhancer.py   — template-based prompt enhancement (no LLM)
  → pipeline/image_gen.py  — SDXL-Turbo (stabilityai/sdxl-turbo, MPS/CUDA/CPU)
  → pipeline/bg_remover.py — rembg u2net background removal → white background
  → pipeline/mesh_gen.py   — Shap-E image300M latent diffusion → PLY → GLB
  → app.py                 — Gradio Model3D viewer displays the .glb
```

**Key design decisions:**
- All pipeline classes (`ImageGenerator`, `BackgroundRemover`, `MeshGenerator`) use **lazy loading** — models are loaded on first use, not at import time. Check `is_loaded()` to inspect state.
- `utils/device.py` is the single source of truth for device selection. Priority: MPS > CUDA > CPU. `MeshGenerator` overrides this and always uses **CPU** because Shap-E has unresolved MPS ops issues.
- `app.py`'s `run_pipeline()` is a **generator function** — it `yield`s 5-tuples `(generated_img, clean_img, glb_path, enhanced_prompt, status_text)` after each stage for incremental Gradio UI updates.
- Pipeline singletons (`image_gen`, `bg_remover`, `mesh_gen`) are module-level globals in `app.py`, shared across Gradio requests.

## Model Details & First-Run Downloads

| Stage | Model | Size | Cache Location |
|---|---|---|---|
| Image gen | `stabilityai/sdxl-turbo` (FP16) | ~6.7 GB | `~/.cache/huggingface/hub/` |
| BG removal | rembg `u2net` | ~170 MB | `~/.u2net/` |
| 3D mesh | Shap-E `image300M` + `transmitter` | ~1.85 GB | `~/.cache/shap_e_model_cache/` |

SDXL-Turbo is a distilled model — `guidance_scale` **must be 0.0** and works best at 4 inference steps. Do not set `guidance_scale > 0` or it degrades output.

## Output Files

Generated meshes are saved to `outputs/` as `mesh_{timestamp}_{seed}.glb`. If GLB export fails, trimesh falls back to `.obj`. Gradio's `Model3D` component accepts both formats.

## Extending the Pipeline

- **Add a new style**: add a key to `STYLE_PRESETS` and `_NEGATIVE_STYLE_EXTRA` in `pipeline/enhancer.py`.
- **Add a new material**: add a key to `MATERIAL_HINTS` in `pipeline/enhancer.py`.
- **Swap the 3D model** (e.g. TripoSR): replace `MeshGenerator` internals in `pipeline/mesh_gen.py`; the `image_to_3d(image) → str` interface must be preserved for `app.py`.
- **Swap the image model**: change `MODEL_ID` in `pipeline/image_gen.py`. If switching away from SDXL-Turbo, also update `guidance_scale` in `app.py`'s `run_pipeline()`.

## Known Constraints (Local / M4)

- 16 GB unified RAM is tight when SDXL-Turbo (~6.5 GB) and Shap-E (~2 GB) are loaded simultaneously. Call `image_gen.unload()` before `mesh_gen.load()` if OOM errors occur.
- `xformers` is not available on MPS; the CUDA-only branch in `image_gen.py` is intentionally skipped on Apple Silicon.
- `invisible-watermark` is listed in requirements but is optional — SDXL safety checker imports it but won't crash without it.
