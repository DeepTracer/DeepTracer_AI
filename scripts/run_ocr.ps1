Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

.\.venv_ocr\Scripts\activate

python.exe trafficsign_storesign_ocr.py `
  --input runs/YOLOn_Trafficsign_Storesign_Track `
  --out runs/YOLOn_Trafficsign_Storesign_OCR `
  --lang korean `
  --rec_model_dir models/korean_PP-OCRv5_mobile_rec
