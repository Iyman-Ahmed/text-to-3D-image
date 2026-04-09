"""
Background removal using rembg (u2net model).

Produces clean RGBA images — critical for Shap-E quality.
A cluttered background causes the 3D model to bake in
background geometry. Removal before 3D gen is highly recommended.

Model: u2net (~170 MB, downloaded on first use to ~/.u2net/)
"""

from __future__ import annotations

from PIL import Image
from typing import Tuple


class BackgroundRemover:
    """Wraps rembg with lazy session loading."""

    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self._session   = None

    # ── Loading ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the rembg session. Idempotent."""
        if self._session is not None:
            return
        from rembg import new_session
        print(f"[BgRemover] Loading rembg ({self.model_name}) …")
        print(f"[BgRemover] First run downloads ~170 MB.")
        self._session = new_session(self.model_name)
        print("[BgRemover] rembg ready.")

    def is_loaded(self) -> bool:
        return self._session is not None

    # ── Core methods ───────────────────────────────────────────────────────

    def remove(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from a PIL image.

        Returns:
            RGBA PIL image with transparent background.
        """
        self.load()
        from rembg import remove
        # rembg accepts both PIL images and bytes
        result = remove(image, session=self._session)
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        return result

    def remove_on_white(self, image: Image.Image) -> Image.Image:
        """
        Remove background and composite onto solid white.
        Preferred input for Shap-E (avoids transparency artefacts).
        """
        rgba = self.remove(image)
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, rgba)
        return composited.convert("RGB")

    def remove_on_color(
        self,
        image: Image.Image,
        bg_rgb: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """Remove background and composite onto a custom solid colour."""
        rgba = self.remove(image)
        background = Image.new("RGBA", rgba.size, bg_rgb + (255,))
        composited = Image.alpha_composite(background, rgba)
        return composited.convert("RGB")
