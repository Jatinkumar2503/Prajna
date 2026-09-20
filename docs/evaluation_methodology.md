# PRAJNA: Evaluation Methodology & Metric Formulations

## 1. Centering Evaluation on $T_{\text{margin}}$

PRAJNA rejects relying solely on classification accuracy. For a nuclear safety decision-support system, the core research metric is the **accuracy and timeliness of remaining margin estimation**:

### 1.1 Margin Prediction Error ($E_T$)
For any time-step $t$, let $T_{\text{margin}}^{\text{ref}}(t)$ denote the true elapsed time until the physical parameter crosses the safety boundary, and $\hat{T}_{\text{margin}}(t)$ denote the model's prediction. The absolute margin error is:
$$E_T(t) = \left| \hat{T}_{\text{margin}}(t) - T_{\text{margin}}^{\text{ref}}(t) \right|$$

Evaluated across the test population $\mathcal{D}_{\text{test}}$:
- **Mean Absolute Error (MAE):** $\text{MAE}_{T} = \frac{1}{N} \sum_{i=1}^N E_T(i)$
- **Median Absolute Error:** $\text{Median}_{T} = \text{Median}(\{E_T(i)\}_{i=1}^N)$
- **95th Percentile Error ($P_{95}$):** $P_{95}(E_T) = \text{Quantile}_{0.95}(\{E_T(i)\}_{i=1}^N)$
- **Maximum Margin Error:** $E_{T,\max} = \max_i E_T(i)$

### 1.2 Early-Warning Lead Time ($\Delta t_{\text{lead}}$)
Let $t_{\text{breach}}$ be the timestamp when the physical parameter violates the provisional safety advisory threshold ($x_t \in \mathcal{S}_{\text{unsafe}}$). Let $t_{\text{alarm}}$ be the timestamp when PRAJNA's safety gate transitions from `NORMAL` to `WARNING` or `CRITICAL advisory`.

The early-warning lead time provided to operators is:
$$\Delta t_{\text{lead}} = t_{\text{breach}} - t_{\text{alarm}}$$
- $\Delta t_{\text{lead}} > 0$: Timely preventative warning before physical threshold breach.
- $\Delta t_{\text{lead}} \le 0$: Late detection (threshold already breached).

### 1.3 Empirical 90% Prediction Interval Coverage
To ensure uncertainty bounds are mathematically trustworthy rather than arbitrary heuristics, prediction intervals are evaluated empirically across all test windows leading up to threshold breaches:
$$\text{Coverage}_{90} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( T_{\text{margin}}^{\text{ref}}(i) \in [\hat{T}_{\text{margin}}(i) - q_{0.90} \sigma(i),\ \hat{T}_{\text{margin}}(i) + q_{0.90} \sigma(i)] \right)$$
where $q_{0.90}$ is the calibrated 90th percentile conformal scale factor.

### 1.4 Lead-Time Specific Margin Error
To prevent misleading global averages that obscure performance when accidents are most critical, margin prediction error $E_T$ is reported at fixed lead times before breach:
$$E_T(\tau) = |\hat{T}_{\text{margin}}(t) - \tau| \quad \text{for } \tau \in \{30\text{s}, 20\text{s}, 10\text{s}, 5\text{s}\}$$

---

## 2. Multi-Step State Forecasting Metrics

Evaluated on multi-step predicted physical trajectories $\hat{X}_{t+1:t+H}$:
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE}_j = \sqrt{\frac{1}{H} \sum_{\tau=1}^H (x_{t+\tau, j} - \hat{x}_{t+\tau, j})^2}$$
- **Mean Absolute Error (MAE):**
  $$\text{MAE}_j = \frac{1}{H} \sum_{\tau=1}^H |x_{t+\tau, j} - \hat{x}_{t+\tau, j}|$$
- **Dynamic Energy Conservation Residual:**
  $$R_{\text{dyn}}(t) = \left| Q_{\text{core}}(t) - \dot{m}(t) C_{p,\text{cool}} (T_{\text{out}}(t) - T_{\text{in}}(t)) - \frac{\Delta E_{\text{stored}}}{\Delta t} \right|$$

---

## 3. Non-Saturated Early Onset Discrimination ($t \le 10\text{ s}$)

To evaluate models without saturation at 100%:
- Evaluates accident identification within $t \le 10\text{ s}$ post-accident initiation under realistic instrument noise ($\sigma_T = \pm 0.5^\circ\text{C}$, $\sigma_P = \pm 0.75\text{ bar}$, ADC quantization).
- Models benchmarked: Logistic Regression, HistGradientBoosting, Rate-of-Change/CUSUM, GRU, LSTM, Temporal Transformer, and PRAJNA Reflex.

---

## 4. Cross-Simulator Per-Unit ($p.u.$) Transformation

To eliminate scaling and temporal rate artifacts when transferring across reactor types (PHWR vs PWR):
1. **Temporal Resampling:** Linearly resample native data to unified $\Delta t = 1.0\text{ s}$.
2. **Per-Unit Normalization:**
   $$x_{\text{p.u.}}(t) = \frac{x(t)}{x_{\text{nominal}}}$$
   ensuring steady-state operational parameters evaluate to $1.0\text{ p.u.}$ across plants.
