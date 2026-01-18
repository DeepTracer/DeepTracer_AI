Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m venv .venv_crop
.\.venv_crop\Scripts\activate

python -m pip install -U pip

# Install torch stack from PyTorch CUDA 12.6 index (exactly as requested)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install the rest from PyPI (exactly as requested)
python -m pip install -r requirements\crop.txt

Write-Host "CROP venv ready."
