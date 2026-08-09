@echo off
cd /d "%~dp0"
title PRAJNA 350M PINN Model Trainer
echo ======================================================================
echo   PRAJNA NUCLEAR INTELLIGENCE — 350M ADVANCED PINN TRAINING ENGINE
echo ======================================================================
echo.
echo Starting 350M parameter PINN model training run...
echo Config: Gradient Checkpointing + Mixed Precision (AMP) + Cosine Annealing
echo.
python scripts/train_pinn_distributed.py --scale advanced_350m --epochs 30 --batch_size 16 --lr 0.0003
pause
