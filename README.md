---
title: Text to 3D Image Generator
emoji: 🧊
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: mit
short_description: Convert any text prompt into a downloadable 3D mesh (GLB/OBJ)
---

# Text-to-3D Image Generator

Convert any text description into a downloadable 3D mesh using a four-stage open-source AI pipeline — no coding required.

---

## Technical Description

**Text-to-3D Generator** is an end-to-end AI pipeline that converts natural language prompts into downloadable 3D mesh files (`.glb` / `.obj`) via four sequential stages:

**1 · Prompt Enhancement**
Template-based keyword injection adds style descriptors, material hints, and 3D-quality suffixes (e.g. *"isolated object, plain white background, 360-degree viewable"*) to improve downstream generation quality. No LLM required — saves 6–8 GB RAM on local deployments.

**2 · Image Generation**
- **HF Spaces (API mode):** Calls the HuggingFace Serverless Inference API with `black-forest-labs/FLUX.1-schnell` — a 4-step distilled diffusion model. Zero model download; just an API call.
- **Local mode:** Loads `stabilityai/sdxl-turbo` (FP16, ~6.7 GB) with MPS / CUDA / CPU backend. Generates at 512×512 in 1–4 steps.

**3 · Background Removal**
`rembg` u2net (ONNX, ~170 MB) removes the background and composites the subject onto white. This isolates the object geometry and prevents Shap-E from baking background noise into the 3D mesh.

**4 · 3D Reconstruction**
OpenAI's Shap-E `image300M` model (~1.5 GB) performs image-conditioned latent diffusion to reconstruct a 3D representation. The output latent is decoded to a triangle mesh via the `transmitter` model (~350 MB), written as PLY, then converted to GLB by `trimesh`. The mesh is normalised to a unit bounding box and centred at origin before display.

**Download sizes:**
| Mode | First-run download |
|---|---|
| HF Spaces (API mode) | ~1.85 GB (Shap-E only) |
| Local | ~8.5 GB (SDXL-Turbo + rembg + Shap-E) |

**Stack:** PyTorch 2.x · Diffusers · Shap-E · rembg · trimesh · Gradio 4.40 · HF Inference API

---

## Local Setup

```bash
git clone https://github.com/Iyman-Ahmed/text-to-3D-Image
cd text-to-3D-Image

# One-shot install (creates venv, installs all deps)
bash install.sh

# Or manually:
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**Verify your environment:**
```bash
python check_env.py
```

**Run the app:**
```bash
source venv/bin/activate
python app.py
# → open http://localhost:7860
```

**Test each pipeline stage individually:**
```bash
python test_pipeline.py --stage 1   # prompt enhancer only (instant)
python test_pipeline.py --stage 2   # + image generation
python test_pipeline.py --stage 3   # + background removal
python test_pipeline.py --stage 4   # + 3D mesh (full pipeline)
```

**Force API mode locally** (no model downloads, needs HF token):
```bash
export HF_TOKEN=hf_...
export USE_API_MODE=1
python app.py
```

---

## HuggingFace Spaces Deployment

This repo is ready to deploy as a Gradio Space:

1. Fork this repo on GitHub
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
3. Select **Gradio SDK**, link your GitHub repo
4. Add a `HF_TOKEN` secret in Space Settings → Repository Secrets
5. The Space auto-detects `SPACE_ID` and switches to API mode (no SDXL download)

---

## Usage Tips

- **Start simple**: single, well-defined objects work best (chair, mug, helmet)
- **Avoid**: scenes, multiple objects, people, abstract concepts
- **Best styles for 3D**: `realistic`, `clay`, `low-poly`
- **Shap-E quality**: increase 3D steps (64→128) for finer detail; takes longer
- **Memory (local)**: close other heavy apps before generating — SDXL-Turbo needs ~6.5 GB free RAM

---

## Project Structure

```
├── app.py                 # Main Gradio app (dual-mode: local + HF)
├── pipeline/
│   ├── enhancer.py        # Template-based prompt enhancement
│   ├── image_gen.py       # SDXL-Turbo (local) + FLUX API (HF)
│   ├── bg_remover.py      # rembg background removal
│   └── mesh_gen.py        # Shap-E image-to-3D + GLB export
├── utils/
│   └── device.py          # MPS / CUDA / CPU device manager
├── check_env.py           # Pre-flight environment checker
├── test_pipeline.py       # Stage-by-stage CLI test runner
├── install.sh             # One-shot local setup script
└── requirements.txt       # Unified deps (HF Spaces + local)
```
