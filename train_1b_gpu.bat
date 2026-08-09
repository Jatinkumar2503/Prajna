@echo off
cd /d "%~dp0"
title PRAJNA 1 BILLION PARAMETER GPU PINN TRAINER
echo ======================================================================
echo   PRAJNA NUCLEAR INTELLIGENCE — 1 BILLION PARAMETER GPU PINN ENGINE
echo ======================================================================
echo.
echo Launching 1,000,000,000 Parameter Foundation Model Training on NVIDIA GPU...
echo Configuration:
echo   - Architecture: 1B Foundation PINN (d_model=2048, 28 Mamba-2 Layers)
echo   - Hardware: NVIDIA CUDA 12.6 + Automatic Mixed Precision (AMP FP16)
echo   - Memory Optimization: Activation Checkpointing + Gradient Accumulation
echo.
"C:\Program Files\Python314\python.exe" scripts/train_pinn_distributed.py --scale foundation_1b --epochs 25 --batch_size 4 --grad_accum 4 --lr 0.0003
pause
