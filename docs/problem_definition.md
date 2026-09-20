# PRAJNA: Formal Problem Definition

## 1. Context & Operational Setting
Nuclear power plants operate within strictly regulated safety margins defined by regulatory authorities (e.g., AERB, US NRC). In a commercial pressurized reactor, hundreds of thermal-hydraulic, neutronic, and mechanical sensors are continuously monitored. 

During an operational transient or developing accident, control room operators and automatic protective interlocks must determine:
1. What will the future state of the core be over the next $H$ seconds?
2. How much time remains before a critical safety boundary is breached?
3. What protective safety action is required?

Traditional industrial monitoring relies on either:
- **Point-wise alarm trip thresholds** (which trigger only after a parameter has already crossed a critical limit, leaving zero preventative lead time), or
- **Statistical anomaly detection models** (which output unconstrained dimensionless scores $[0, 1]$ that suffer from high false alarm rates during normal operational power maneuvers).

PRAJNA formulates this challenge as **physics-constrained temporal state forecasting coupled to deterministic safety-boundary margin estimation**.

---

## 2. Mathematical Problem Formulation

### Input: Historical Sensor Telemetry
Let the discrete time index be $t \in \mathbb{N}$ with sampling interval $\Delta t = 1.0\text{ s}$. The historical observation window of length $W$ is denoted:
$$X_{t-W+1:t} = [x_{t-W+1}, x_{t-W+2}, \dots, x_t] \in \mathbb{R}^{W \times D}$$
where $D = 12$ physical sensor channels (Core Exit Temperature, Primary Flow, Neutron Flux, Radiation, Primary Pressure, Core Power, Control Rod Position, Pressurizer Level, Steam Generator Temp, Steam Flow, Core Inlet Temp, Containment Pressure).

### Output 1: Multi-Step State Trajectory Forecasting
Given $X_{t-W+1:t}$, predict the future physical trajectory over forward horizon $H$:
$$\hat{X}_{t+1:t+H} = [\hat{x}_{t+1}, \hat{x}_{t+2}, \dots, \hat{x}_{t+H}] \in \mathbb{R}^{H \times D}$$
subject to dynamic energy and momentum conservation constraints:
$$\mathcal{L}_{\text{dyn}}(\hat{X}_{t+1:t+H}) \le \epsilon_{\text{phys}}$$

### Output 2: Deterministic Safety Margin ($T_{\text{margin}}$)
Let $\mathcal{S}_{\text{unsafe}} \subset \mathbb{R}^D$ denote the licensed unsafe operational domain defined by regulatory limits (e.g., $T_{\text{out}} \ge 305.0^\circ\text{C}$, $P_{\text{prim}} \ge 93.0\text{ bar}$ or $\le 72.0\text{ bar}$, $\phi_{\text{th}} \ge 2.55 \times 10^{13}\text{ n/cm}^2\cdot\text{s}$).

The remaining time to threshold breach $T_{\text{margin}}$ is defined as:
$$T_{\text{margin}}(t) = \min_{\tau > 0} \left\{ \tau \cdot \Delta t \;\middle|\; \hat{x}_{t+\tau} \in \mathcal{S}_{\text{unsafe}} \right\}$$
with upper bound $T_{\text{margin}} = H$ if no threshold crossing is projected within horizon $H$.

### Output 3: Deterministic Safety Gating
Downstream of the forecasted trajectory and $T_{\text{margin}}$, the plant safety state $S(t)$ is evaluated deterministically:
$$S(t) = \begin{cases}
\text{NORMAL}, & \text{if } T_{\text{margin}}(t) > \tau_{\text{warn}} \text{ and } \hat{x}_{t} \notin \mathcal{S}_{\text{unsafe}} \\
\text{WARNING}, & \text{if } \tau_{\text{crit}} < T_{\text{margin}}(t) \le \tau_{\text{warn}} \\
\text{CRITICAL\_TRIP}, & \text{if } T_{\text{margin}}(t) \le \tau_{\text{crit}} \text{ or } x_t \in \mathcal{S}_{\text{unsafe}}
\end{cases}$$
where $\tau_{\text{warn}} = 30.0\text{ s}$ and $\tau_{\text{crit}} = 15.0\text{ s}$.
