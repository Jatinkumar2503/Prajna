@echo off
cd /d "%~dp0"
title PRAJNA 125M PINN Model Trainer
echo ======================================================================
echo   PRAJNA NUCLEAR INTELLIGENCE — 125M PARAMETER PINN TRAINING ENGINE
echo ======================================================================
echo.
echo Starting high-efficiency multi-physics training run (125 Million Parameters)...
echo Config: Mixed Precision (AMP) + Cosine Annealing + Physics Loss Regularizer
echo.
python scripts/train_pinn_production.py --scale efficient_125m --epochs 50 --batch_size 32 --lr 0.0005
pause
