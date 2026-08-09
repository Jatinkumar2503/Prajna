# PRAJNA 2.25B PINN Training Launcher (PowerShell)
Set-Location -Path $PSScriptRoot
$host.UI.RawUI.WindowTitle = "PRAJNA 2.25B PINN Model Trainer"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  PRAJNA NUCLEAR INTELLIGENCE — 2.25B PINN FOUNDATION MODEL TRAINER" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Launching 2,250 Million parameter PINN model training run..." -ForegroundColor Yellow
Write-Host "Config: Activation Checkpointing + Mixed Precision (AMP) + Physics Loss" -ForegroundColor Gray
Write-Host ""

python scripts/train_pinn_distributed.py --scale intermediate_2.25b --epochs 25 --batch_size 8 --lr 0.0002

Write-Host ""
Write-Host "Training process completed." -ForegroundColor Green
Read-Host "Press Enter to exit..."
