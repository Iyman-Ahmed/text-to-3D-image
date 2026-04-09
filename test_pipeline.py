#!/usr/bin/env python3
"""
Standalone pipeline smoke test — runs outside Gradio.

Usage:
    python test_pipeline.py [--stage 1|2|3|4]

Stages:
    1  Prompt enhancer only       (no model download)
    2  Prompt + SDXL-Turbo image  (~6.7 GB download on first run)
    3  Stage 2 + rembg BG remove  (~170 MB download on first run)
    4  All stages incl. Shap-E    (~1.85 GB download on first run)

Running stage 1 first is instant and confirms imports work.
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GREEN = "\033[32m"
RED   = "\033[31m"
BLUE  = "\033[34m"
RESET = "\033[0m"

def step(n, msg): print(f"\n{BLUE}[{n}]{RESET} {msg}")
def ok(msg):      print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg):    print(f"  {RED}✗{RESET}  {msg}")


def test_stage_1():
    """Prompt enhancement — zero downloads, instant."""
    step(1, "Prompt enhancement")
    from pipeline.enhancer import enhance_prompt, get_negative_prompt

    prompt   = "a simple wooden chair"
    enhanced = enhance_prompt(prompt, "realistic", "wood")
    neg      = get_negative_prompt("realistic")

    ok(f"Enhanced ({len(enhanced)} chars)")
    print(f"     → {enhanced[:100]} …")
    ok(f"Negative  ({len(neg)} chars)")


def test_stage_2():
    """SDXL-Turbo image generation."""
    step(2, "Image generation  (SDXL-Turbo)")
    from pipeline.image_gen import ImageGenerator
    from pipeline.enhancer import enhance_prompt, get_negative_prompt

    prompt   = enhance_prompt("a simple wooden chair", "realistic", "wood")
    neg      = get_negative_prompt("realistic")

    gen = ImageGenerator()
    ok(f"Device: {gen.device}  dtype: {gen.dtype}")

    print("  Loading model (may download ~6.7 GB on first run) …")
    t0  = time.time()
    img = gen.generate(prompt=prompt, negative_prompt=neg, num_steps=4, seed=42)
    elapsed = time.time() - t0

    out_path = "outputs/test_stage2.png"
    os.makedirs("outputs", exist_ok=True)
    img.save(out_path)
    ok(f"Image saved → {out_path}  [{img.size[0]}×{img.size[1]}]  ({elapsed:.1f}s)")
    return img


def test_stage_3(image=None):
    """Background removal."""
    step(3, "Background removal  (rembg u2net)")
    from pipeline.bg_remover import BackgroundRemover
    from PIL import Image

    if image is None:
        path = "outputs/test_stage2.png"
        if not os.path.exists(path):
            fail(f"No image at {path} — run stage 2 first")
            sys.exit(1)
        image = Image.open(path)

    remover = BackgroundRemover()
    print("  Loading rembg (may download ~170 MB on first run) …")
    t0    = time.time()
    clean = remover.remove_on_white(image)
    elapsed = time.time() - t0

    out_path = "outputs/test_stage3_clean.png"
    clean.save(out_path)
    ok(f"Clean image saved → {out_path}  ({elapsed:.1f}s)")
    return clean


def test_stage_4(image=None):
    """Shap-E 3D generation."""
    step(4, "3D mesh generation  (Shap-E)")
    from pipeline.mesh_gen import MeshGenerator
    from PIL import Image

    if image is None:
        path = "outputs/test_stage3_clean.png"
        if not os.path.exists(path):
            path = "outputs/test_stage2.png"
        if not os.path.exists(path):
            fail(f"No input image — run stages 2 and 3 first")
            sys.exit(1)
        image = Image.open(path)

    gen = MeshGenerator()
    ok(f"Device: {gen.device}")
    print("  Loading Shap-E (may download ~1.85 GB on first run) …")
    print("  Running 32 diffusion steps (reduced for quick test) …")
    t0       = time.time()
    glb_path = gen.image_to_3d(image, num_steps=32, guidance_scale=3.0, seed=42)
    elapsed  = time.time() - t0

    ok(f"GLB saved → {glb_path}  ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Text-to-3D pipeline smoke test")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3, 4], default=1,
        help="Which stage to run up to (default: 1)"
    )
    args = parser.parse_args()

    print("=" * 58)
    print(f"  Text-to-3D Pipeline Test  —  up to stage {args.stage}")
    print("=" * 58)

    try:
        test_stage_1()
        if args.stage >= 2:
            img = test_stage_2()
        if args.stage >= 3:
            img = test_stage_3(img if args.stage >= 2 else None)
        if args.stage == 4:
            test_stage_4(img if args.stage >= 3 else None)

        print(f"\n{GREEN}All stages up to {args.stage} passed.{RESET}\n")

    except Exception as e:
        import traceback
        fail(f"Stage failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
