"""
3D mesh generation using Shap-E (OpenAI).

Pipeline: PIL Image → Shap-E latent diffusion → TriMesh → GLB file

Model: image300M (~1.5 GB) + transmitter (~350 MB)
Downloaded on first use to ~/.cache/shap_e_model_cache/

Notes:
  - Shap-E MPS support is experimental; we default to CPU for reliability.
  - use_fp16=False on CPU (FP16 requires CUDA/MPS tensor support in Shap-E).
  - GLB is exported via trimesh for Gradio's Model3D component.
"""

from __future__ import annotations

import os
import gc
import tempfile
import time
import torch
import numpy as np
from PIL import Image
from typing import Optional, Callable


class MeshGenerator:
    """Wraps Shap-E image-to-3D pipeline with lazy loading."""

    def __init__(self, device: str = None):
        from utils.device import get_device
        raw_device = device or get_device()
        # Shap-E's MPS support has known issues with some ops.
        # Default to CPU for correctness; fast enough for local testing.
        self.device = "cpu" if raw_device == "mps" else raw_device
        self._xm        = None
        self._model     = None
        self._diffusion = None

    # ── Loading ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load Shap-E models into memory. Idempotent."""
        if self._xm is not None:
            return

        from shap_e.models.download import load_model, load_config
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config

        print(f"[MeshGen] Loading Shap-E on {self.device} …")
        print(f"[MeshGen] First run downloads ~1.85 GB from HuggingFace.")

        self._xm        = load_model("transmitter",  device=self.device)
        self._model     = load_model("image300M",    device=self.device)
        self._diffusion = diffusion_from_config(load_config("diffusion"))

        print("[MeshGen] Shap-E ready.")

    def is_loaded(self) -> bool:
        return self._xm is not None

    # ── Core method ────────────────────────────────────────────────────────

    def image_to_3d(
        self,
        image: Image.Image,
        num_steps: int = 64,
        guidance_scale: float = 3.0,
        seed: int = 0,
        output_dir: str = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Convert a PIL image to a 3D mesh saved as a GLB file.

        Args:
            image:             Input image (white background recommended).
            num_steps:         Diffusion steps (16–128; 64 is good default).
            guidance_scale:    3.0–5.0 typical range.
            seed:              For reproducibility.
            output_dir:        Directory to write the GLB. Defaults to outputs/.
            progress_callback: Optional fn(current_step, total_steps).

        Returns:
            Absolute path to the generated .glb file.
        """
        self.load()

        from shap_e.diffusion.sample import sample_latents

        # Prepare image (Shap-E expects 256×256 RGB)
        img = image.convert("RGB").resize((256, 256), Image.LANCZOS)

        # Output path
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "outputs"
            )
        os.makedirs(output_dir, exist_ok=True)
        timestamp  = int(time.time())
        glb_path   = os.path.join(output_dir, f"mesh_{timestamp}_{seed}.glb")
        ply_path   = glb_path.replace(".glb", ".ply")

        print(f"[MeshGen] Sampling latents ({num_steps} steps, "
              f"guidance={guidance_scale}) …")

        latents = sample_latents(
            batch_size=1,
            model=self._model,
            diffusion=self._diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(images=[img]),
            progress=True,               # tqdm in terminal
            clip_denoised=True,
            use_fp16=False,              # FP16 unsupported on CPU
            use_karras=True,
            karras_steps=num_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        print("[MeshGen] Decoding latent to mesh …")
        with torch.no_grad():
            # decode_latent_mesh from shap_e.util.notebooks is just xm.decode(latent[None])
            # but that module imports ipywidgets which isn't available outside Jupyter
            tri_mesh = self._xm.decode(latents[0][None]).tri_mesh()

        # Write PLY (Shap-E native format)
        with open(ply_path, "wb") as f:
            tri_mesh.write_ply(f)

        # Convert PLY → GLB via trimesh
        glb_path = self._ply_to_glb(ply_path, glb_path)

        # Clean up intermediate PLY
        try:
            os.remove(ply_path)
        except OSError:
            pass

        print(f"[MeshGen] Saved GLB → {glb_path}")
        return glb_path

    # ── Conversion helper ──────────────────────────────────────────────────

    @staticmethod
    def _ply_to_glb(ply_path: str, glb_path: str) -> str:
        """
        Convert a PLY file to GLB using trimesh.
        Returns the actual output path (may fall back to .obj).
        """
        import trimesh

        mesh = trimesh.load(ply_path, process=False)

        # trimesh may return a Scene for PLYs with vertex colours
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()]
            )

        # Normalise scale to fit in a unit bounding box
        if mesh.vertices is not None and len(mesh.vertices) > 0:
            bounds = mesh.bounding_box.extents
            max_extent = max(bounds)
            if max_extent > 0:
                mesh.apply_scale(1.0 / max_extent)
            # Centre at origin
            mesh.apply_translation(-mesh.centroid)

        try:
            mesh.export(glb_path)
            return glb_path
        except Exception as e:
            # GLB export failed — fall back to OBJ (also supported by Gradio)
            print(f"[MeshGen] GLB export failed ({e}), trying OBJ fallback …")
            obj_path = glb_path.replace(".glb", ".obj")
            mesh.export(obj_path)
            return obj_path

    # ── Memory management ──────────────────────────────────────────────────

    def unload(self) -> None:
        """Release all Shap-E model weights."""
        self._xm        = None
        self._model     = None
        self._diffusion = None
        gc.collect()
        print("[MeshGen] Shap-E unloaded.")
