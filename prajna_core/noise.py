"""
PRAJNA REALISTIC SENSOR DEGRADATION & INSTRUMENT NOISE SUITE
Implements empirical instrumentation noise and degradation effects:
1. Gaussian measurement noise (RTDs, SPNDs, Piezoelectric transducers)
2. Calibration bias and thermal drift ramps
3. Industrial ADC resolution quantization
4. First-order thermowell and instrument response lag (tau)
5. Channel dropouts, frozen/stuck-at values, and EMI transients
"""

import math
import torch
from typing import Dict, Optional, Tuple

# Standard nuclear instrumentation measurement uncertainties (1-sigma) for 756 MWth PHWR
INSTRUMENT_SIGMAS = torch.tensor([
    0.50,   # 0: Core Exit Temp: ±0.50 °C (Class A Pt100 RTD)
    5.00,   # 1: Coolant Flow: ±5.0 kg/s (Venturi differential pressure, ~0.14% of 3700 kg/s)
    0.015,  # 2: Neutron Flux: ±0.015 x10^13 (SPND detector)
    0.025,  # 3: Radiation: ±0.025 mSv/h (Gamma Ion Chamber)
    0.75,   # 4: Primary Pressure: ±0.75 bar (Piezoelectric transmitter)
    3.50,   # 5: Core Power: ±3.50 MWth (Calorimetric balance, ~0.46% of 756 MWth)
    0.40,   # 6: Control Rod Height: ±0.40 % (Digital synchro-resolver)
    0.50,   # 7: Pressurizer Level: ±0.50 % (Differential pressure cell)
    0.60,   # 8: Feedwater Temp: ±0.60 °C (Secondary RTD)
    2.50,   # 9: Steam Flow: ±2.50 kg/s (Vortex shedding flowmeter, ~0.24% of 1050 kg/s)
    0.50,   # 10: Core Inlet Temp: ±0.50 °C (Cold leg RTD)
    0.40,   # 11: Containment Pressure: ±0.40 kPa (Capacitive sensor)
    0.50,   # 12: Steam Gen Level: ±0.50 % (Secondary differential pressure)
    0.50,   # 13: Core Delta-P: ±0.50 kPa (Header differential pressure)
    0.50,   # 14: Loop Flow: ±0.50 % (Primary loop balance)
    0.20    # 15: Turbine Speed: ±0.20 % (Digital tachometer)
], dtype=torch.float32)

# ADC quantization step sizes (LSB resolution)
ADC_RESOLUTIONS = torch.tensor([
    0.1, 0.2, 0.005, 0.005, 0.1, 0.5, 0.001, 0.2, 0.2, 0.1, 0.2, 0.1, 0.05, 0.2, 0.005, 0.1
], dtype=torch.float32)

# Thermowell / sensor first-order response time constants (seconds)
# Thermocouples in wells: tau = 2.5 - 4.0s; electronic pressure < 0.1s;
# SPND: Platinum/Cobalt (prompt gamma) tau ~ 0.1s; Rhodium beta-decay component has tau ~ 42s.
# We model prompt SPND as default (0.1s) with optional delayed Rhodium emitter lag (42.0s).
SENSOR_TAU = torch.tensor([
    3.0, 0.2, 0.1, 0.5, 0.1, 1.0, 2.0, 0.1, 0.8, 3.0, 0.3, 3.0, 3.0, 2.5, 0.1, 0.5
], dtype=torch.float32)


def apply_instrument_noise_suite(x: torch.Tensor,
                                 enable_lag: bool = True,
                                 enable_quantization: bool = True,
                                 enable_drift: bool = True,
                                 enable_dropouts: bool = True,
                                 noise_scale: float = 1.0) -> torch.Tensor:
    """
    Applies realistic instrument degradation across a batch of reactor telemetry.
    x: [Batch, SeqLen, Channels]
    Returns degraded telemetry tensor of same shape.
    """
    batch_size, seq_len, num_channels = x.shape
    device = x.device
    sigmas = INSTRUMENT_SIGMAS[:num_channels].to(device) * noise_scale
    out = x.clone()

    # 1. First-Order Sensor Lag: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    # alpha = dt / (tau + dt) with dt = 1.0s
    if enable_lag and seq_len > 1:
        taus = SENSOR_TAU[:num_channels].to(device)
        alphas = 1.0 / (taus + 1.0)  # [Channels]
        lagged = torch.zeros_like(out)
        lagged[:, 0, :] = out[:, 0, :]
        for t in range(1, seq_len):
            lagged[:, t, :] = alphas * out[:, t, :] + (1.0 - alphas) * lagged[:, t - 1, :]
        out = lagged

    # 2. Gaussian Measurement Noise (Independent per channel)
    gaussian_noise = torch.randn_like(out) * sigmas.view(1, 1, -1)
    out = out + gaussian_noise

    # 3. Calibration Drift Ramps (Slow linear bias over sequence)
    if enable_drift:
        drift_rates = (torch.rand(batch_size, 1, num_channels, device=device) - 0.5) * 0.002
        time_steps = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, -1, 1)
        out = out + drift_rates * time_steps

    # 4. ADC Quantization (Rounding to finite industrial bit resolution)
    if enable_quantization:
        res = ADC_RESOLUTIONS[:num_channels].to(device)
        out = torch.round(out / res) * res

    # 5. Sparse Transients & Stuck-At Dropouts (1% probability of temporary sensor freeze)
    if enable_dropouts and seq_len > 5:
        # Stuck-at fault mask: freeze a channel for last 5 samples with 2% probability
        stuck_mask = torch.rand(batch_size, 1, num_channels, device=device) < 0.02
        if stuck_mask.any():
            frozen_val = out[:, -6:-5, :].expand(-1, 5, -1)
            out[:, -5:, :] = torch.where(stuck_mask, frozen_val, out[:, -5:, :])

    return out


def apply_stuck_sensor_fault(x: torch.Tensor,
                             channels: Optional[list] = None,
                             start_step: int = 10) -> torch.Tensor:
    """
    Simulates frozen/stuck-at sensor transmitter failure.
    From start_step onward, specified channels hold their exact previous value.
    """
    out = x.clone()
    batch_size, seq_len, num_channels = out.shape
    if channels is None:
        channels = [4]  # Primary pressure sensor by default
    if start_step >= seq_len:
        return out
    
    frozen_val = out[:, start_step - 1:start_step, :]
    for ch in channels:
        if ch < num_channels:
            out[:, start_step:, ch] = frozen_val[:, 0, ch].unsqueeze(-1).expand(-1, seq_len - start_step)
    return out


def apply_step_bias_fault(x: torch.Tensor,
                          channels: Optional[list] = None,
                          biases: Optional[list] = None,
                          start_step: int = 5) -> torch.Tensor:
    """
    Simulates abrupt sensor calibration step bias (e.g. electrical surge or transducer shift).
    """
    out = x.clone()
    batch_size, seq_len, num_channels = out.shape
    if channels is None:
        channels = [4]
    if biases is None:
        biases = [2.5]
    if start_step >= seq_len:
        return out

    for ch, bias in zip(channels, biases):
        if ch < num_channels:
            out[:, start_step:, ch] = out[:, start_step:, ch] + bias
    return out


def apply_deadband_hysteresis(x: torch.Tensor,
                              thresholds: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Simulates mechanical deadband / hysteresis in measurement transducers.
    Transmitter only updates if physical change exceeds deadband threshold.
    """
    batch_size, seq_len, num_channels = x.shape
    device = x.device
    if thresholds is None:
        thresholds = (INSTRUMENT_SIGMAS[:num_channels] * 0.8).to(device)
    else:
        thresholds = thresholds[:num_channels].to(device)

    out = torch.zeros_like(x)
    out[:, 0, :] = x[:, 0, :]
    for t in range(1, seq_len):
        delta = x[:, t, :] - out[:, t - 1, :]
        update_mask = torch.abs(delta) >= thresholds.view(1, -1)
        out[:, t, :] = torch.where(update_mask, x[:, t, :], out[:, t - 1, :])
    return out


def inject_sensor_fault_suite(x: torch.Tensor,
                              fault_prob: float = 0.35,
                              seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, list]]:
    """
    Randomly injects realistic sensor instrumentation faults across a batch:
    - Stuck/frozen transducer (stuck-at fault)
    - Abrupt step calibration bias (transducer shift)
    - Deadband / stiction hysteresis (mechanical hysteresis)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    out = x.clone()
    batch_size, seq_len, num_channels = out.shape
    fault_log: Dict[str, list] = {"stuck": [], "bias": [], "deadband": []}
    
    for b in range(batch_size):
        r = float(torch.rand(1).item())
        if r < fault_prob * 0.4:
            ch = int(torch.randint(0, num_channels, (1,)).item())
            step = int(torch.randint(max(1, seq_len // 4), max(2, seq_len * 3 // 4), (1,)).item())
            out[b:b+1] = apply_stuck_sensor_fault(out[b:b+1], channels=[ch], start_step=step)
            fault_log["stuck"].append((b, ch, step))
        elif r < fault_prob * 0.8:
            ch = int(torch.randint(0, num_channels, (1,)).item())
            step = int(torch.randint(max(1, seq_len // 4), max(2, seq_len * 3 // 4), (1,)).item())
            sigma_val = float(INSTRUMENT_SIGMAS[ch].item()) if ch < len(INSTRUMENT_SIGMAS) else 1.0
            bias_val = float((torch.rand(1).item() * 2 - 1) * 3.0 * sigma_val)
            out[b:b+1] = apply_step_bias_fault(out[b:b+1], channels=[ch], biases=[bias_val], start_step=step)
            fault_log["bias"].append((b, ch, bias_val, step))
        elif r < fault_prob:
            out[b:b+1] = apply_deadband_hysteresis(out[b:b+1])
            fault_log["deadband"].append(b)
            
    return out, fault_log

