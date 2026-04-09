#!/usr/bin/env python3
"""
Pre-flight environment check for Text-to-3D Generator.

Run BEFORE app.py to validate your setup:
    python check_env.py

Checks:
  - Python version
  - All required packages
  - PyTorch device (MPS / CUDA / CPU)
  - Disk space (models need ~10 GB on first run)
  - Available RAM
  - HuggingFace connectivity
"""

import sys
import os
import importlib
import shutil
from typing import Tuple, List

RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"

def ok(msg):  return f"{GREEN}✓{RESET}  {msg}"
def err(msg): return f"{RED}✗{RESET}  {msg}"
def warn(msg):return f"{YELLOW}!{RESET}  {msg}"


# ── Individual checks ─────────────────────────────────────────────────────────

def check_python() -> Tuple[bool, str]:
    v = sys.version_info
    version_str = f"Python {v.major}.{v.minor}.{v.micro}"
    if v < (3, 9):
        return False, f"{version_str}  (3.9+ required)"
    if v < (3, 10):
        return True, f"{version_str}  (3.10+ recommended for best compatibility)"
    return True, version_str


def check_torch() -> Tuple[bool, str, str]:
    """Returns (ok, message, device_str)."""
    try:
        import torch
        ver = torch.__version__
        if torch.backends.mps.is_available():
            # Quick MPS smoke test
            try:
                x = torch.zeros(1, device="mps")
                _ = x + 1
                device = "mps"
                device_label = f"Apple Silicon GPU (MPS)  [PyTorch {ver}]"
            except Exception as e:
                device = "cpu"
                device_label = f"MPS available but failed ({e}) — using CPU  [PyTorch {ver}]"
        elif torch.cuda.is_available():
            device = "cuda"
            device_label = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})  [PyTorch {ver}]"
        else:
            device = "cpu"
            device_label = f"CPU only  [PyTorch {ver}]"
        return True, device_label, device
    except ImportError:
        return False, "PyTorch NOT installed", "none"


def check_package(pkg_name: str, import_name: str = None) -> Tuple[bool, str]:
    imp = import_name or pkg_name
    try:
        mod = importlib.import_module(imp)
        ver = getattr(mod, "__version__", "?")
        return True, f"{pkg_name}=={ver}"
    except ImportError:
        return False, f"{pkg_name}  NOT installed"


def check_shap_e() -> Tuple[bool, str]:
    try:
        from shap_e.models.download import load_model  # noqa: F401
        import shap_e
        ver = getattr(shap_e, "__version__", "installed")
        return True, f"shap-e  {ver}"
    except ImportError:
        return False, "shap-e  NOT installed  →  pip install shap-e"


def check_disk(required_gb: float = 12.0) -> Tuple[bool, str]:
    home = os.path.expanduser("~")
    total, used, free = shutil.disk_usage(home)
    free_gb = free / (1024 ** 3)
    status = free_gb >= required_gb
    note = "" if status else f"  (need ~{required_gb:.0f} GB for first-run model downloads)"
    return status, f"{free_gb:.1f} GB free{note}"


def check_ram(required_gb: float = 8.0) -> Tuple[bool, str]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb  = mem.total     / (1024 ** 3)
        avail_gb  = mem.available / (1024 ** 3)
        status    = avail_gb >= required_gb
        note = "" if status else f"  (need ~{required_gb:.0f} GB free; SDXL-Turbo uses ~6.5 GB)"
        return status, f"{avail_gb:.1f} GB available / {total_gb:.1f} GB total{note}"
    except ImportError:
        return True, "psutil not found — RAM check skipped"


def check_hf_connectivity() -> Tuple[bool, str]:
    try:
        import requests
        r = requests.head("https://huggingface.co", timeout=5)
        return r.status_code < 400, "huggingface.co reachable"
    except Exception as e:
        return False, f"huggingface.co unreachable ({e}) — models need internet on first run"


def check_outputs_dir() -> Tuple[bool, str]:
    path = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(path, exist_ok=True)
    return True, f"outputs/ directory ready  ({path})"


# ── Pipeline module import checks ─────────────────────────────────────────────

def check_pipeline_modules() -> List[Tuple[bool, str]]:
    results = []
    modules = [
        ("utils.device",       "utils/device.py"),
        ("pipeline.enhancer",  "pipeline/enhancer.py"),
        ("pipeline.image_gen", "pipeline/image_gen.py"),
        ("pipeline.bg_remover","pipeline/bg_remover.py"),
        ("pipeline.mesh_gen",  "pipeline/mesh_gen.py"),
    ]
    sys.path.insert(0, os.path.dirname(__file__))
    for mod, path in modules:
        try:
            importlib.import_module(mod)
            results.append((True, f"{path}  importable"))
        except ImportError as e:
            results.append((False, f"{path}  import error: {e}"))
        except Exception as e:
            results.append((False, f"{path}  error: {e}"))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{BOLD}{'='*62}{RESET}")
    print(f"{BOLD}   Text-to-3D Generator — Environment Check{RESET}")
    print(f"{BOLD}{'='*62}{RESET}\n")

    has_errors = False
    missing_pkgs = []

    # 1. Python
    ok_flag, msg = check_python()
    print(f"  {ok(msg) if ok_flag else warn(msg)}")
    if not ok_flag:
        has_errors = True

    # 2. Torch + device
    print()
    torch_ok, torch_msg, device = check_torch()
    print(f"  {'PyTorch':<22} {ok(torch_msg) if torch_ok else err(torch_msg)}")
    if not torch_ok:
        has_errors = True
        missing_pkgs.append("torch")

    # 3. Core packages
    print()
    packages = [
        ("diffusers",       "diffusers"),
        ("transformers",    "transformers"),
        ("accelerate",      "accelerate"),
        ("safetensors",     "safetensors"),
        ("gradio",          "gradio"),
        ("rembg",           "rembg"),
        ("onnxruntime",     "onnxruntime"),
        ("trimesh",         "trimesh"),
        ("pygltflib",       "pygltflib"),
        ("PIL",             "PIL"),
        ("numpy",           "numpy"),
        ("scipy",           "scipy"),
        ("huggingface_hub", "huggingface_hub"),
        ("requests",        "requests"),
        ("tqdm",            "tqdm"),
        ("psutil",          "psutil"),
    ]
    for name, imp in packages:
        flag, msg = check_package(name, imp)
        print(f"  {name:<22} {ok(msg) if flag else err(msg)}")
        if not flag:
            has_errors = True
            missing_pkgs.append(name)

    # 4. Shap-E (separate because install is slightly different)
    print()
    flag, msg = check_shap_e()
    print(f"  {'shap-e':<22} {ok(msg) if flag else err(msg)}")
    if not flag:
        has_errors = True
        missing_pkgs.append("shap-e")

    # 5. System resources
    print()
    disk_ok, disk_msg = check_disk()
    ram_ok,  ram_msg  = check_ram()
    print(f"  {'Disk space':<22} {ok(disk_msg) if disk_ok else warn(disk_msg)}")
    print(f"  {'RAM available':<22} {ok(ram_msg) if ram_ok else warn(ram_msg)}")

    # 6. Connectivity
    print()
    net_ok, net_msg = check_hf_connectivity()
    print(f"  {'HuggingFace':<22} {ok(net_msg) if net_ok else warn(net_msg)}")

    # 7. Outputs dir
    _, out_msg = check_outputs_dir()
    print(f"  {'outputs/':<22} {ok(out_msg)}")

    # 8. Pipeline module imports
    print()
    print(f"  {BOLD}Pipeline modules:{RESET}")
    for flag, msg in check_pipeline_modules():
        print(f"    {ok(msg) if flag else err(msg)}")

    # Summary
    print(f"\n{BOLD}{'='*62}{RESET}")
    if missing_pkgs:
        print(f"\n  {RED}{BOLD}Missing:{RESET}  {', '.join(missing_pkgs)}")
        print(f"\n  Install all dependencies:")
        print(f"    pip install -r requirements.txt")
        if "shap-e" in missing_pkgs:
            print(f"    pip install shap-e")
        print()
    else:
        print(f"\n  {GREEN}{BOLD}All checks passed!{RESET}")
        print(f"  Device: {device.upper()}")
        print(f"\n  First-run note: models (~10 GB) download automatically")
        print(f"  from HuggingFace on first pipeline execution.\n")
        print(f"  Start the app:")
        print(f"    python app.py\n")

    return 0 if not has_errors else 1


if __name__ == "__main__":
    sys.exit(main())
