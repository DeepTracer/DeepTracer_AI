Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python -m venv .venv_ocr
.\.venv_ocr\Scripts\activate

python -m pip install -U pip setuptools wheel

# Install Paddle GPU from Paddle index (exactly as requested)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
python -c "import paddle; paddle.utils.run_check(); print('Paddle OK')"

# Install PaddleOCR version pin via requirements (same result as your command)
python -m pip install -r requirements\ocr.txt
python -c "from paddleocr import PaddleOCR; ocr=PaddleOCR(lang='korean'); print('PaddleOCR init OK')"

Write-Host "OCR venv ready."
