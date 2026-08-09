# PRAJNA 1 Billion Parameter PINN GPU Training Launcher (PowerShell)
Set-Location -Path $PSScriptRoot
$host.UI.RawUI.WindowTitle = "PRAJNA 1B GPU PINN Model Trainer"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  PRAJNA NUCLEAR INTELLIGENCE — 1 BILLION PARAMETER GPU PINN ENGINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching 1,000,000,000 Parameter Model Training on NVIDIA GPU..." -ForegroundColor Yellow
Write-Host "Config: CUDA 12.6 + FP16 Mixed Precision (AMP) + Activation Checkpointing" -ForegroundColor Gray
Write-Host ""

python scripts/train_pinn_distributed.py --scale foundation_1b --epochs 25 --batch_size 4 --grad_accum 4 --lr 0.0003

Write-Host ""
Write-Host "Training process completed." -ForegroundColor Green
Read-Host "Press Enter to exit..."
