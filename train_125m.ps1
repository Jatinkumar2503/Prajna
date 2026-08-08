# PRAJNA 125M PINN Model Trainer (PowerShell)
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  PRAJNA NUCLEAR INTELLIGENCE — 125M PARAMETER PINN TRAINING ENGINE" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting high-efficiency multi-physics training run (125 Million Parameters)..." -ForegroundColor Yellow
Write-Host "Config: Mixed Precision (AMP) + Cosine Annealing + Physics Loss Regularizer" -ForegroundColor Gray
Write-Host ""

python scripts/train_pinn_production.py --scale efficient_125m --epochs 50 --batch_size 32 --lr 0.0005
