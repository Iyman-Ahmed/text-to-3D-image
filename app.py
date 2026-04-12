"""
Text-to-3D Generator
=====================
Dual-mode: local models OR HuggingFace Inference API (no download).

Local run:
    python app.py

HuggingFace Spaces:
    Deployed automatically — image generation uses HF Serverless
    Inference API (FLUX.1-schnell), Shap-E runs in the Space container.

Pipeline:
    Text prompt
      → Prompt enhancer  (template-based, no LLM)
      → Image generation (SDXL-Turbo local  OR  FLUX.1-schnell via API)
      → rembg            (background removal, CPU)
      → Shap-E           (image-to-3D, CPU, ~1.85 GB on first run)
      → Gradio Model3D viewer
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from utils.device import get_device
from pipeline.enhancer import (
    enhance_prompt,
    get_negative_prompt,
    list_styles,
    list_materials,
)
from pipeline.image_gen   import ImageGenerator
from pipeline.bg_remover  import BackgroundRemover
from pipeline.mesh_gen    import MeshGenerator

# ── Environment detection ──────────────────────────────────────────────────────
# SPACE_ID is set automatically by HuggingFace Spaces runtime
IS_HF_SPACE = bool(os.environ.get("SPACE_ID"))

# API mode: use HF Serverless Inference for image gen (zero local download).
# Automatically enabled on HF Spaces; can override with USE_API_MODE=1 locally.
USE_API = IS_HF_SPACE or os.environ.get("USE_API_MODE", "0") == "1"

# ── Global pipeline singletons (lazy-loaded) ──────────────────────────────────
DEVICE    = get_device()
image_gen = ImageGenerator(use_api=USE_API)
bg_remover = BackgroundRemover()
mesh_gen  = MeshGenerator()

DEVICE_LABELS = {
    "mps":  "Apple Silicon GPU (MPS)",
    "cuda": "NVIDIA CUDA GPU",
    "cpu":  "CPU",
    "api":  "HF Serverless Inference API",
}

# ── Pipeline function ─────────────────────────────────────────────────────────

def run_pipeline(
    prompt:          str,
    style:           str,
    material:        str,
    img_steps:       int,
    mesh_steps:      int,
    seed:            int,
    guidance_3d:     float,
    remove_bg:       bool,
):
    """
    Full text-to-3D pipeline with streaming Gradio progress.
    Yields (generated_image, clean_image, glb_path, enhanced_prompt, status)
    after each stage so the UI updates incrementally.
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    log_lines: list[str] = []

    def log(msg: str) -> str:
        log_lines.append(msg)
        return "\n".join(log_lines)

    try:
        # ── Step 1: Enhance prompt ─────────────────────────────────────────
        enhanced  = enhance_prompt(prompt, style, material)
        neg_prompt = get_negative_prompt(style)
        status = log("[1/4] ✓ Prompt enhanced")
        yield None, None, None, enhanced, status

        # ── Step 2: Generate image ─────────────────────────────────────────
        status = log(f"[2/4] ⟳ Generating image ({img_steps} steps, SDXL-Turbo) …\n"
                     f"      (first run downloads ~6.7 GB — this may take a few minutes)")
        yield None, None, None, enhanced, status

        generated_image = image_gen.generate(
            prompt=enhanced,
            negative_prompt=neg_prompt,
            num_steps=img_steps,
            guidance_scale=0.0,   # SDXL-Turbo distilled — must be 0.0
            seed=seed,
        )

        status = log("[2/4] ✓ Image generated")
        yield generated_image, None, None, enhanced, status

        # ── Step 3: Remove background ──────────────────────────────────────
        if remove_bg:
            status = log("[3/4] ⟳ Removing background (rembg u2net) …\n"
                         "      (first run downloads ~170 MB)")
            yield generated_image, None, None, enhanced, status

            clean_image = bg_remover.remove_on_white(generated_image)

            status = log("[3/4] ✓ Background removed")
            yield generated_image, clean_image, None, enhanced, status
        else:
            clean_image = generated_image
            status = log("[3/4] — Background removal skipped")
            yield generated_image, clean_image, None, enhanced, status

        # ── Step 4: Generate 3D mesh ───────────────────────────────────────
        status = log(f"[4/4] ⟳ Building 3D mesh ({mesh_steps} steps, Shap-E) …\n"
                     f"      (first run downloads ~1.85 GB — please wait)")
        yield generated_image, clean_image, None, enhanced, status

        glb_path = mesh_gen.image_to_3d(
            image=clean_image,
            num_steps=mesh_steps,
            guidance_scale=guidance_3d,
            seed=max(seed, 0),
        )

        status = log("[4/4] ✓ 3D mesh saved\n\n🎉 Pipeline complete!")
        yield generated_image, clean_image, glb_path, enhanced, status

    except Exception as e:
        import traceback
        err_msg = f"❌ Error: {e}\n\n{traceback.format_exc()}"
        status = log(err_msg)
        yield None, None, None, enhanced if 'enhanced' in dir() else "", status


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    if USE_API:
        mode_tag  = "HuggingFace Spaces · API Mode"
        img_label = "FLUX.1-schnell (HF API — no download)"
    else:
        mode_tag  = "Local · " + DEVICE_LABELS.get(DEVICE, DEVICE.upper())
        img_label = "SDXL-Turbo (local)"

    with gr.Blocks(
        title="Text-to-3D Generator",
        theme=gr.themes.Soft(primary_hue="violet"),
        css="""
            .gradio-container { max-width: 1280px !important; margin: auto; }
            .status-box textarea { font-family: monospace; font-size: 0.85rem; }
        """,
    ) as demo:

        gr.Markdown(f"""
# Text-to-3D Generator  ·  {mode_tag}
**Pipeline**: Text → {img_label} → Background Removal → Shap-E → 3D Model
**Device**: {DEVICE_LABELS.get(DEVICE, DEVICE.upper())}
> Models download once at startup (~2 GB). Subsequent runs are fast.
        """)

        with gr.Row():
            # ── LEFT: inputs ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="a wooden chair with ornate carved legs",
                    lines=3,
                )

                with gr.Row():
                    style_input = gr.Dropdown(
                        label="Style",
                        choices=list_styles(),
                        value="realistic",
                    )
                    material_input = gr.Dropdown(
                        label="Material",
                        choices=list_materials(),
                        value="default",
                    )

                remove_bg_toggle = gr.Checkbox(
                    label="Remove background before 3D conversion (recommended)",
                    value=True,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    seed_input = gr.Slider(
                        label="Seed  (-1 = random)",
                        minimum=-1, maximum=9999, step=1, value=42,
                    )
                    img_steps_input = gr.Slider(
                        label="Image steps  (SDXL-Turbo: 1–8)",
                        minimum=1, maximum=8, step=1, value=4,
                    )
                    mesh_steps_input = gr.Slider(
                        label="3D steps  (Shap-E: 8–64, higher = better)",
                        minimum=8, maximum=64, step=8, value=16,
                    )
                    guidance_3d_input = gr.Slider(
                        label="3D guidance scale",
                        minimum=1.0, maximum=10.0, step=0.5, value=3.0,
                    )

                generate_btn = gr.Button(
                    "Generate 3D", variant="primary", size="lg"
                )

                status_box = gr.Textbox(
                    label="Pipeline Status",
                    lines=7,
                    interactive=False,
                    elem_classes=["status-box"],
                )

                enhanced_prompt_box = gr.Textbox(
                    label="Enhanced Prompt (sent to SDXL-Turbo)",
                    lines=5,
                    interactive=False,
                )

            # ── RIGHT: outputs ─────────────────────────────────────────────
            with gr.Column(scale=1, min_width=420):

                with gr.Tabs():
                    with gr.Tab("Generated Image"):
                        generated_img_out = gr.Image(
                            label="SDXL-Turbo output",
                            type="pil",
                            height=350,
                        )

                    with gr.Tab("After BG Removal"):
                        clean_img_out = gr.Image(
                            label="Input to Shap-E",
                            type="pil",
                            height=350,
                        )

                model_3d_out = gr.Model3D(
                    label="3D Model  (drag to rotate)",
                    height=420,
                )

        # ── Examples ──────────────────────────────────────────────────────
        gr.Examples(
            label="Quick-start examples",
            examples=[
                ["a wooden chair with ornate carved legs",       "realistic", "wood"],
                ["a futuristic battle helmet",                   "sci-fi",    "metal"],
                ["a cute cartoon baby dragon",                   "cartoon",   "default"],
                ["a simple ceramic coffee mug",                  "realistic", "ceramic"],
                ["a smooth polished river stone",                "realistic", "stone"],
                ["a low-poly fox sitting on its tail",           "low-poly",  "default"],
            ],
            inputs=[prompt_input, style_input, material_input],
        )

        # ── Wire up button ─────────────────────────────────────────────────
        generate_btn.click(
            fn=run_pipeline,
            inputs=[
                prompt_input,
                style_input,
                material_input,
                img_steps_input,
                mesh_steps_input,
                seed_input,
                guidance_3d_input,
                remove_bg_toggle,
            ],
            outputs=[
                generated_img_out,
                clean_img_out,
                model_3d_out,
                enhanced_prompt_box,
                status_box,
            ],
            api_name=False,   # disables schema introspection that triggers gradio_client bug
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = "HuggingFace Spaces (API)" if IS_HF_SPACE else "Local"
    print("=" * 60)
    print(f"  Text-to-3D Generator  —  {mode}")
    print(f"  Image mode : {'HF Inference API (FLUX.1-schnell)' if USE_API else 'Local SDXL-Turbo'}")
    if not IS_HF_SPACE:
        print(f"  Device     : {DEVICE_LABELS.get(DEVICE, DEVICE)}")
    print("=" * 60)

    # Pre-load Shap-E on startup so the first user request isn't slow.
    # On HF Spaces this downloads ~1.85 GB once during the boot phase.
    print("  Pre-loading Shap-E models …")
    try:
        mesh_gen.load()
        print("  Shap-E ready.")
    except Exception as e:
        print(f"  Warning: Shap-E pre-load failed ({e}) — will retry on first request.")

    demo = build_ui()
    demo.queue()   # required for generator / streaming functions

    if IS_HF_SPACE:
        # HuggingFace Spaces manages host/port automatically
        demo.launch()
    else:
        print("\n  Open your browser at:  http://localhost:7860")
        print("  Stop the server with:  Ctrl+C\n")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
        )
