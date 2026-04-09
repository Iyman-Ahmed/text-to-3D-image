#!/bin/bash
# ── Text-to-3D Generator — One-shot setup (macOS / Apple Silicon) ─────────────
set -e

echo ""
echo "========================================================"
echo "  Text-to-3D Generator — Installation"
echo "========================================================"
echo ""

# 1. Confirm Python 3.9+
PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python $PY detected"

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment …"
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Upgrade pip (3.9 ships with old pip)
echo "  Upgrading pip …"
pip install --upgrade pip --quiet

# 4. Install PyTorch (Apple Silicon native build — no CUDA needed)
echo "  Installing PyTorch (MPS-enabled) …"
pip install torch torchvision torchaudio --quiet

# 5. Install all other deps
echo "  Installing project dependencies …"
pip install -r requirements.txt --quiet

# 6. Install Shap-E from GitHub (not available on PyPI)
echo "  Installing Shap-E …"
pip install "git+https://github.com/openai/shap-e.git" --quiet

# 7. Run pre-flight check
echo ""
echo "  Running environment check …"
echo ""
python check_env.py

echo ""
echo "========================================================"
echo "  Setup complete!"
echo ""
echo "  Activate the environment and run the app:"
echo "    source venv/bin/activate"
echo "    python app.py"
echo "========================================================"
echo ""
