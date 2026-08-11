# PRAJNA 1 Billion Parameter PINN GPU Training Launcher (PowerShell)
Set-Location -Path $PSScriptRoot
$host.UI.RawUI.WindowTitle = "PRAJNA 1B GPU PINN Model Trainer"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  PRAJNA NUCLEAR INTELLIGENCE — 1 BILLION PARAMETER GPU PINN ENGINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching 1,023,878,496 Parameter Full Foundation Model Training on NVIDIA GPU..." -ForegroundColor Yellow
Write-Host "Config: CUDA + FP16 Mixed Precision (AMP) + Activation Checkpointing + Gradient Accumulation (2 x 8)" -ForegroundColor Gray
Write-Host ""

python scripts/train_pinn_distributed.py --scale full_1b --epochs 15 --batch_size 2 --grad_accum 8 --lr 0.0002

Write-Host ""
Write-Host "Training process completed." -ForegroundColor Green
Read-Host "Press Enter to exit..."
