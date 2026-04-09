"""
Template-based prompt enhancer for the Text-to-3D pipeline.

No LLM required — saves ~6-8 GB RAM for the local version.
Adds quality, style, and 3D-relevant keywords to user prompts.
"""

from typing import Tuple

# ── Style presets ─────────────────────────────────────────────────────────────

STYLE_PRESETS: dict = {
    "realistic": (
        "photorealistic, highly detailed, 8k resolution, "
        "professional product photography, studio lighting, sharp focus"
    ),
    "cartoon": (
        "cartoon style, vibrant colors, clean bold lines, "
        "cel-shaded, 3D render, Pixar style"
    ),
    "low-poly": (
        "low poly art style, geometric facets, minimal, "
        "clean edges, game asset, flat shading"
    ),
    "clay": (
        "clay render, matte plasticine material, soft diffuse shadows, "
        "pastel tones, Octane render, smooth surface"
    ),
    "sci-fi": (
        "sci-fi industrial design, futuristic, hard surface modeling, "
        "metallic panels, glowing elements, concept art"
    ),
    "organic": (
        "organic sculpted form, natural material, ZBrush detail, "
        "subsurface scattering, nature-inspired"
    ),
}

# ── Material hints ────────────────────────────────────────────────────────────

MATERIAL_HINTS: dict = {
    "default": "",
    "metal":   "metallic surface, brushed steel, high specularity, reflective",
    "wood":    "natural wood grain, warm amber tones, carved surface, matte finish",
    "stone":   "rough stone texture, weathered rock, grey tones, carved marble",
    "glass":   "transparent glass material, refractive, crystal clear, clean",
    "fabric":  "woven fabric texture, soft cloth material, detailed weave pattern",
    "ceramic": "ceramic glaze, smooth rounded surface, matte or satin finish",
    "plastic": "smooth plastic surface, injection mold, clean finish, product design",
}

# ── Universal 3D-quality suffix ───────────────────────────────────────────────

_QUALITY_SUFFIX = (
    "single isolated object, centered composition, plain white background, "
    "360-degree viewable, clean silhouette, no shadows on background"
)

_NEGATIVE_BASE = (
    "blurry, low quality, low resolution, multiple objects, busy background, "
    "text, watermark, signature, frame, border, deformed, disfigured, ugly"
)

_NEGATIVE_STYLE_EXTRA: dict = {
    "realistic": ", cartoon, illustration, drawing, painting, sketch",
    "cartoon":   ", photorealistic, photo, real",
    "low-poly":  ", photorealistic, smooth surface, high poly",
    "clay":      ", metallic, glossy, photorealistic",
    "sci-fi":    ", organic, natural, rustic, wooden",
    "organic":   ", mechanical, metallic, geometric",
}


# ── Public API ────────────────────────────────────────────────────────────────

def enhance_prompt(
    prompt: str,
    style: str = "realistic",
    material: str = "default",
) -> str:
    """
    Build an enhanced prompt for SDXL-Turbo from user input.

    Args:
        prompt:   Raw user text, e.g. "a wooden chair with carved legs"
        style:    Key from STYLE_PRESETS
        material: Key from MATERIAL_HINTS

    Returns:
        Enhanced prompt string ready for image generation.
    """
    parts = [prompt.strip().rstrip(",")]

    style_text = STYLE_PRESETS.get(style, STYLE_PRESETS["realistic"])
    parts.append(style_text)

    material_text = MATERIAL_HINTS.get(material, "")
    if material_text:
        parts.append(material_text)

    parts.append(_QUALITY_SUFFIX)

    return ", ".join(p for p in parts if p)


def get_negative_prompt(style: str = "realistic") -> str:
    """Return a style-appropriate negative prompt."""
    extra = _NEGATIVE_STYLE_EXTRA.get(style, "")
    return _NEGATIVE_BASE + extra


def list_styles() -> list:
    return list(STYLE_PRESETS.keys())


def list_materials() -> list:
    return list(MATERIAL_HINTS.keys())


def preview_enhancement(
    prompt: str,
    style: str = "realistic",
    material: str = "default",
) -> Tuple[str, str]:
    """Returns (enhanced_prompt, negative_prompt) for display."""
    return enhance_prompt(prompt, style, material), get_negative_prompt(style)
