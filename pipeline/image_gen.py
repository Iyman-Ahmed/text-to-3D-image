"""
Image generation — dual mode:

  LOCAL  (default for local dev)
    Model:   stabilityai/sdxl-turbo  (~6.7 GB FP16)
    Device:  MPS / CUDA / CPU
    Steps:   1–8  (4 recommended)

  API  (default on HuggingFace Spaces — zero local download)
    Model:   black-forest-labs/FLUX.1-schnell  (HF Serverless Inference)
    Needs:   HF_TOKEN env var (auto-set inside HF Spaces)
    Steps:   4 (handled server-side)

Set use_api=True to force API mode regardless of environment.
"""

from __future__ import annotations

import os
import torch
from PIL import Image
from typing import Optional, Callable

LOCAL_MODEL_ID = "stabilityai/sdxl-turbo"
API_MODEL_ID   = "black-forest-labs/FLUX.1-schnell"


class ImageGenerator:
    """Wraps SDXL-Turbo (local) or FLUX.1-schnell (HF API) with lazy loading."""

    def __init__(
        self,
        device: str = None,
        dtype: torch.dtype = None,
        use_api: bool = False,
    ):
        self.use_api = use_api
        self._pipe   = None

        if not use_api:
            from utils.device import get_device, get_dtype
            self.device = device or get_device()
            self.dtype  = dtype  or get_dtype(self.device)
        else:
            self.device = "api"
            self.dtype  = None

    # ── Public API ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load local SDXL-Turbo. No-op in API mode."""
        if self.use_api or self._pipe is not None:
            return

        from diffusers import AutoPipelineForText2Image

        print(f"[ImageGen] Loading SDXL-Turbo → {self.device} ({self.dtype}) …")
        print("[ImageGen] First run downloads ~6.7 GB from HuggingFace.")

        self._pipe = AutoPipelineForText2Image.from_pretrained(
            LOCAL_MODEL_ID,
            torch_dtype=self.dtype,
            variant="fp16",
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()

        if self.device == "cuda":
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        print("[ImageGen] SDXL-Turbo ready.")

    def is_loaded(self) -> bool:
        return self.use_api or (self._pipe is not None)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        """
        Generate an image.  Routes to API or local depending on mode.

        Args:
            prompt:           Enhanced text prompt.
            negative_prompt:  Ignored in API mode (FLUX.1-schnell is distilled).
            num_steps:        Local only. API handles steps server-side.
            guidance_scale:   0.0 for SDXL-Turbo; 3.5 typical for FLUX local.
            seed:             -1 = random.
            width / height:   512×512 default (local). API uses 512×512 too.
            callback:         Local only — fn(step, total).
        """
        if self.use_api:
            return self._api_generate(prompt, seed=seed, width=width, height=height)
        return self._local_generate(
            prompt, negative_prompt, num_steps,
            guidance_scale, seed, width, height, callback
        )

    def unload(self) -> None:
        if self._pipe is not None:
            from utils.device import free_memory
            free_memory(self._pipe)
            self._pipe = None
            print("[ImageGen] SDXL-Turbo offloaded.")

    # ── Private: local inference ────────────────────────────────────────────

    def _local_generate(
        self,
        prompt, negative_prompt, num_steps,
        guidance_scale, seed, width, height, callback,
    ) -> Image.Image:
        self.load()

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        callback_on_step = None
        if callback is not None:
            def callback_on_step(pipe, step_index, timestep, callback_kwargs):
                callback(step_index + 1, num_steps)
                return callback_kwargs

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=callback_on_step,
        )
        return result.images[0]

    # ── Private: HF Serverless Inference API ────────────────────────────────

    def _api_generate(
        self,
        prompt: str,
        seed: int = -1,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """
        Call HF Serverless Inference API for FLUX.1-schnell.
        Free tier — no local model download needed.
        Requires HF_TOKEN in env (auto-available inside HF Spaces).
        """
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        client = InferenceClient(model=API_MODEL_ID, token=token)

        print(f"[ImageGen] Calling HF Inference API ({API_MODEL_ID}) …")

        kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=4,
        )
        if seed >= 0:
            kwargs["seed"] = seed

        image = client.text_to_image(**kwargs)

        if not isinstance(image, Image.Image):
            from io import BytesIO
            image = Image.open(BytesIO(image))

        print("[ImageGen] API image received.")
        return image
