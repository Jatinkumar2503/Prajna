# PRAJNA 350M PINN Training Launcher (PowerShell)
Set-Location -Path $PSScriptRoot
$host.UI.RawUI.WindowTitle = "PRAJNA 350M PINN Model Trainer"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  PRAJNA NUCLEAR INTELLIGENCE — 350M ADVANCED PINN TRAINING ENGINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching 350M parameter PINN model training run..." -ForegroundColor Yellow
Write-Host "Config: Gradient Checkpointing + Mixed Precision (AMP) + Cosine Annealing" -ForegroundColor Gray
Write-Host ""

python scripts/train_pinn_distributed.py --scale advanced_350m --epochs 30 --batch_size 16 --lr 0.0003

Write-Host ""
Write-Host "Training process completed." -ForegroundColor Green
Read-Host "Press Enter to exit..."
