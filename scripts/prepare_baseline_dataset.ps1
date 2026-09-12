# End-to-end baseline dataset prep: download CropAndWeed -> YOLO pool -> 80/10/10 split
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . ".venv\Scripts\Activate.ps1"
}

Write-Host "=== Step 1/3: Download CropAndWeed ==="
python -m src.training.download_cropandweed @args

Write-Host ""
Write-Host "=== Step 2/3: Convert to YOLO segmentation pool ==="
python -m src.training.convert_cropandweed

Write-Host ""
Write-Host "=== Step 3/3: Random 80/10/10 split ==="
python -m src.training.split_dataset

Write-Host ""
Write-Host "Baseline dataset ready at data/yolo_dataset/"
Write-Host "Next: python -m src.training.train --config configs/train_config.yaml"
