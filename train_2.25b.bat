@echo off
cd /d "%~dp0"
title PRAJNA 2.25B PINN Foundation Model Trainer
echo ======================================================================
echo   PRAJNA NUCLEAR INTELLIGENCE — 2.25 BILLION PARAMETER PINN TRAINER
echo ======================================================================
echo.
echo Starting 2,250 Million parameter PINN foundation model training run...
echo Config: Activation Checkpointing + Mixed Precision (AMP) + Physics Loss
echo.
python scripts/train_pinn_distributed.py --scale intermediate_2.25b --epochs 25 --batch_size 8 --lr 0.0002
pause
