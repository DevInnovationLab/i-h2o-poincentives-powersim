# PO Incentives Power Simulation

Statistical power analysis for a staggered rollout randomized controlled trial (RCT) measuring the effect of financial incentives on pump operator (PO) chlorination behavior across 40 villages in Andhra Pradesh.

---

## 1. Study Design

### 1.1 Overview

The intervention installs Inline Chlorine (ILC) devices at 40 village water points in Andhra Pradesh and evaluates whether paying pump operators for verified chlorination increases chlorine presence in the water supply. The study uses a staggered rollout design where villages are enrolled over time, followed by a within-village randomization to treatment (payments) or control (monitoring only).

### 1.2 Timeline

Each village passes through four phases, staggered by its installation date:

| Phase | Relative Weeks | Description |
|-------|---------------|-------------|
| **Installation** | Week 0 | ILC device installed at village water point |
| **Stabilization** | Weeks 1–4 | Equipment settles in; no monitoring or data collection |
| **Training & Monitoring** | Weeks 5–8 | PO trained on digital self-reporting app; independent chlorine measurements begin (default 2x/week); this serves as the **pre-treatment baseline** |
| **Treatment Period** | Weeks 9 onward (variable duration) | Random half of villages begin receiving payments for chlorine presence; monitoring continues for all villages. Duration depends on installation date and study end week (6 months or 1 year from AP start). |

### 1.3 Installation Schedule

Villages are enrolled at a constant rate over 8 calendar weeks:

- **Calendar weeks 1–8:** 5 villages installed per week
- **Total:** 5 × 8 = 40 villages

Because installation is staggered, each village's phases occur at different calendar times. A village installed in calendar week 1 begins treatment at calendar week 9, while a village installed in calendar week 8 begins treatment at calendar week 16.

### 1.4 Treatment Assignment

At the start of each village's treatment period (relative week 9), villages are randomly assigned:
- **20 villages → Treatment group:** PO receives financial payments conditional on chlorine being detected in independent measurements.
- **20 villages → Control group:** PO continues using the self-reporting app with independent monitoring, but receives no payments.

### 1.5 Outcome Measurement

Each week during the monitoring and treatment periods, 2 independent chlorine measurements are taken at the water point. Each measurement is binary (chlorine detected or not). The weekly outcome is the **proportion of positive measurements**:

```
Y_it = (1/K) · Σ m_j ∈ {0, 1/K, 2/K, ..., 1}
```

where m_j ∈ {0, 1} for j = 1, ..., K and K is the number of measurements per week.

---

## 2. Data Generating Process (DGP)

### 2.1 Site-Level Baseline Compliance

Each pump operator has an intrinsic propensity to add chlorine, drawn from a truncated normal distribution:

```
θ_i ~ TruncNormal(μ_baseline, σ_baseline, 0, 1)
```

- `μ_baseline` (Baseline Compliance Rate): The average propensity across all POs. Sweeping this parameter captures uncertainty about how often POs chlorinate without incentives.
- `σ_baseline` (Compliance Heterogeneity): How much POs vary in their baseline behavior. Higher values mean some POs almost always chlorinate while others almost never do.

### 2.2 Weekly Behavioral Model

Each week, the PO's effective propensity to chlorinate follows an AR(1) (first-order autoregressive) process:

```
p_it = clip[(1 - ρ) · θ_i + ρ · Y_{i,t-1} + τ · D_it + h(t) · M_it, 0, 1]
```

where:
- `p_it` is PO i's probability of chlorinating in week t
- `θ_i` is the PO's baseline propensity (drawn once, fixed for the study)
- `Y_{i,t-1}` is the previous week's observed outcome (proportion of positive measurements), initialized to θ_i for the first observed week
- `ρ` (Behavioral Persistence): AR(1) coefficient controlling how much last week's behavior influences this week. Higher ρ means behavior is more "sticky" — a PO who chlorinated last week is more likely to chlorinate this week.
- `τ` (per-period treatment impulse): The direct weekly effect of the payment incentive on the propensity to chlorinate. Only applied when `D_it = 1` (treated village in the treatment period).
- `h(t)` (Monitoring/Hawthorne Effect): A time-varying effect of being monitored (see Section 2.3).
- `M_it = 1` whenever the site is in the monitoring window (training or treatment phase).
- `clip[·, 0, 1]` constrains the propensity to valid probability bounds.

**Why AR(1)?** Pump operator behavior is unlikely to be independent week-to-week. A PO who chlorinated last week may have established a routine, purchased supplies, or simply formed a habit. The AR(1) process captures this behavioral persistence. The parameter ρ controls the strength: ρ = 0 means fully independent decisions each week; ρ = 0.9 means behavior is highly persistent and slow to change.

**Measurement process:** Given propensity p_it, the K weekly measurements are independent Bernoulli draws:

```
m_j ~ Bernoulli(p_it)   for j = 1, ..., K
Y_it = (1/K) · Σ m_j
```

The noisy outcome Y_it (not the latent propensity p_it) feeds back into the AR(1) process. This means measurement noise propagates through the behavioral dynamics, which is realistic: the PO observes whether he actually chlorinated (not his latent propensity), and that observation influences next week's behavior.

### 2.3 Time-Varying Monitoring (Hawthorne) Effect

The act of being monitored (self-reporting app + independent measurements) may itself change PO behavior. This could go in either direction:

- **Positive h_init (e.g., +0.10):** POs initially increase chlorination when they realize they're being watched, but this novelty effect fades over time.
- **Negative h_init (e.g., -0.10):** POs initially resist or are confused by the new monitoring system, leading to temporarily lower chlorination, but they adapt over time.

The Hawthorne effect decays linearly over the full monitoring window:

```
h(t) = h_init · max(0, 1 - (relative_week - 5) / T_monitoring)
```

where `T_monitoring = 52` weeks (4 training + 48 treatment). At the start of monitoring (relative week 5), the effect equals `h_init`. It decays linearly toward zero over the monitoring period.

**Importantly**, the Hawthorne effect applies equally to treated and control villages (both are monitored). The difference-in-differences estimator removes its level effect, but because villages are installed at different times, the Hawthorne is at different decay stages for different villages at the same calendar week. This creates a subtle interaction with the staggered design.

### 2.4 Dynamic Treatment Effect and the AR(1) Amplification

The per-period impulse `τ` is **not** the same as the treatment effect the estimator recovers. Because of the AR(1) feedback, the treatment effect accumulates over time:

- **Week 0 of treatment:** Effect = τ
- **Week 1:** Effect = τ + ρ · τ = τ(1 + ρ)
- **Week 2:** Effect = τ(1 + ρ + ρ²)
- **Week k:** Effect = τ · Σ_{j=0}^{k} ρ^j = τ · (1 - ρ^{k+1}) / (1 - ρ)

The **average treatment effect** over T treatment weeks (the estimand our DiD recovers) is:

```
ATT_avg = τ · [T - ρ(1 - ρ^T)/(1 - ρ)] / [T · (1 - ρ)]
```

This "amplification factor" depends on both ρ and T:

| ρ | Steady-state amplification (T→∞) | 48-week amplification | 16-week amplification |
|---|---|---|---|
| 0.0 | 1.0× | 1.0× | 1.0× |
| 0.5 | 2.0× | 1.96× | 1.88× |
| 0.7 | 3.3× | 3.10× | 2.73× |
| 0.9 | 10.0× | 7.09× | 4.57× |

**Reparameterization:** Rather than sweeping τ (which has different implications for different ρ values), we sweep `target_att` — the desired average effect on the chlorination rate. The simulation back-calculates τ from target_att using the finite-horizon formula:

```
τ = target_att / amplification_factor
```

This means "target_att = 0.10" always represents a 10 percentage point increase in the chlorination rate, regardless of ρ. The simulation adjusts the per-period impulse accordingly.

---

## 3. Estimation: Difference-in-Differences

### 3.1 Estimator

We use a site-level difference-in-differences estimator. For each site, we collapse the panel to a single pre/post score:

```
δ_i = mean(Y_i in treatment phase) - mean(Y_i in training phase)
```

The estimated ATT is:

```
ATT_hat = mean(δ_i for treated sites) - mean(δ_i for control sites)
```

This is equivalent to a Callaway & Sant'Anna (2021) estimator in the special case where the control group is "never-treated" (control sites never receive payments). The site-level collapse ensures that within-site serial correlation is handled by construction — each site contributes one independent observation.

### 3.2 Standard Errors

We use Welch's two-sample formula, which provides cluster-robust standard errors at the site level:

```
SE = sqrt(Var(δ_treated) / n_treated + Var(δ_control) / n_control)
```

where `Var(δ)` is the sample variance of the site-level scores within each group, computed with Bessel's correction (ddof=1).

This is correct because:
1. Each δ_i is a single independent observation at the cluster (site) level.
2. Treatment assignment is random and independent across sites.
3. The two-sample formula allows for unequal variances between groups.

### 3.3 Hypothesis Test

We test the sharp null H₀: ATT = 0 using Welch's t-test (unequal variance two-sample t-test):

```
t = ATT_hat / SE
Reject H₀ if p-value < 0.05 (two-sided)
```

The t-test uses Satterthwaite degrees of freedom, which is more appropriate than a z-test when each arm has only 20 clusters.

### 3.4 Verification

Under the null (target_att = 0), this estimator produces:
- **Rejection rate ≈ 5%** (correct size) across all ρ values
- **SE/SD ratio ≈ 1.0** (standard errors match actual sampling variability)
- **Mean ATT ≈ 0** (unbiased)

Under the alternative (target_att > 0), the mean estimated ATT closely matches the target across all parameter configurations.

---

## 4. Power Analysis

### 4.1 Simulation Design

For each combination of parameters, we:
1. Generate a simulated panel dataset from the DGP
2. Estimate the ATT and compute the standard error
3. Record whether the null hypothesis is rejected

This is repeated 1,000 times per parameter combination. **Power** is the proportion of simulations that reject the null.

### 4.2 Parameter Sweep

All sweep ranges are defined in **`sweep_params.csv`**, a single CSV file that serves as the source of truth for every parameter grid. Edit this file to adjust the sweep ranges to your preferences — the code reads it at runtime.

The CSV has four columns:

| Column | Purpose |
|--------|---------|
| `parameter` | Parameter name (used by code) |
| `values` | Comma-separated list of values to sweep |
| `description` | Human-readable explanation of the parameter |
| `unit` | Unit of measurement |

Default sweep ranges:

| Parameter | Description | Values | Count |
|-----------|-------------|--------|-------|
| `mu_baseline` | Baseline Compliance Rate | 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 | 6 |
| `target_att` | Target Effect on Chlorination Rate | 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40 | 10 |
| `rho` | Behavioral Persistence (AR1) | 0.5, 0.7, 0.9 | 3 |

Cross-site heterogeneity (`sigma_baseline`) is derived from `mu_baseline`: σ = 0.8 × √(μ(1−μ)).

**Sweep: 6 × 10 × 3 = 180 parameter combinations × 2 durations = 360 combos × 1,000 simulations = 360,000 total simulations.**

### 4.3 Key Output: Minimum Detectable Effect (MDE)

The primary output is the **MDE at 80% power** — the smallest target_att value for which the design achieves at least 80% power, for each combination of baseline compliance, behavioral persistence, compliance heterogeneity, and monitoring effect.

For example, an MDE of 0.10 means: "With 40 villages (20 treated, 20 control), we can detect a 10 percentage point increase in chlorination rates with 80% probability."

---

## 5. Usage

### 5.1 Installation

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib, seaborn, tqdm.

### 5.2 Running the Simulation

```bash
# Run power sweep and output MDE tables
python run_simulation.py --n_sims 1000

# Quick test (fewer sims, faster):
python run_simulation.py --n_sims 100
```

Output:
- MDE tables printed to console (rows = baseline compliance, columns = AR(1) persistence)
- `output/power_results.csv` — raw power estimates for all parameter combos
- `output/mde_table_6_months.csv` — MDE table for 6-month duration
- `output/mde_table_1_year.csv` — MDE table for 1-year duration

### 5.3 HPC Usage (SLURM)

For full precision (1,000 sims), use the HPC cluster. The included `submit_hpc.sh` is configured for the UChicago RCC MidwaySSD partition:

```bash
sbatch submit_hpc.sh
```

---

## 6. Output Files

| File | Description |
|------|-------------|
| `output/power_results.csv` | Power estimates for all parameter combos and both durations |
| `output/mde_table_6_months.csv` | MDE at 80% power for 6-month study |
| `output/mde_table_1_year.csv` | MDE at 80% power for 1-year study |

---

## 7. File Structure

| File | Purpose |
|------|---------|
| `sweep_params.csv` | **Single source of truth** for all parameter sweep ranges — edit this to change what gets swept |
| `config.py` | Loads `sweep_params.csv`, defines parameter grids, constants, installation schedules |
| `generate_data.py` | Data generating process — staggered rollout panel generation |
| `estimate.py` | Difference-in-differences estimator |
| `run_simulation.py` | **Main entry point** — runs power sweep and outputs MDE tables |
| `run_power_sweep.py` | Lower-level parallel sweep with `--hpc` mode |
| `visualize.py` | Plots and MDE tables from power results |
| `submit_hpc.sh` | SLURM job submission script for UChicago RCC |
| `requirements.txt` | Python dependencies |

---

## 8. Key Design Decisions

1. **No TWFE.** Two-way fixed effects regressions are biased under staggered treatment timing (Goodman-Bacon 2021, de Chaisemartin & D'Haultfoeuille 2020). We use a site-level DiD that is equivalent to Callaway & Sant'Anna (2021) with a never-treated control group.

2. **Site-level collapse for SEs.** Rather than computing influence function-based clustered SEs (which are error-prone for complex aggregated estimators), we collapse the panel to one pre/post score per site. This eliminates within-site serial correlation by construction and yields a simple, provably correct two-sample SE.

3. **Finite-horizon reparameterization.** The parameter sweep is expressed in terms of the target dynamic ATT (the actual effect on chlorination rates the estimator recovers), not the per-period behavioral impulse τ. The per-period impulse is back-calculated using the finite-horizon AR(1) amplification formula, accounting for the specific treatment duration.

4. **Noisy AR(1) feedback.** The AR(1) process feeds back the observed outcome Y (average of 2 Bernoulli draws), not the latent propensity p. This is behaviorally realistic: the PO observes his actual chlorination behavior, not his latent inclination.
