# Create virtual environment and install dependencies (Windows / PowerShell).
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    & $Python -m venv .venv
}

$Activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
. $Activate

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run tests: pytest tests/"
Write-Host "Prepare baseline dataset:"
Write-Host "  python -m src.training.download_cropandweed"
Write-Host "  python -m src.training.convert_cropandweed"
Write-Host "  python -m src.training.split_dataset"
Write-Host "Run live pipeline:"
Write-Host "  python -m src.inference.live_pipeline --config configs/pipeline_config.yaml"
