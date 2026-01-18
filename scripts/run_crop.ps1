Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

.\.venv_crop\Scripts\activate

python.exe trafficsign_storesign_crop.py `
  --model runs/YOLOn_Trafficsign_Storesign_Train/weights/best.pt `
  --source ../input.mp4 `
  --tracker runs/YOLOn_Trafficsign_Storesign_Train/botsort.yaml `
  --imgsz 640 `
  --conf 0.5 `
  --iou 0.5 `
  --device 0 `
  --out_root runs/YOLOn_Trafficsign_Storesign_Track `
  --save_meta `
  --use_obb_warp
