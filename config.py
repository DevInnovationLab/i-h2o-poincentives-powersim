"""Shared configuration for the power simulation."""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
N_MEASUREMENTS = 2          # independent chlorine measurements per week
STUDY_END_WEEK = 52         # calendar week when data collection ends (1 year from AP start)
N_SIMS = 1000
SEED = 42

# Phase durations (in weeks, relative to installation)
STABILIZATION_WEEKS = 4
TRAINING_WEEKS = 4
TRAINING_REL_START = STABILIZATION_WEEKS + 1  # relative week 5
TREATMENT_REL_START = TRAINING_REL_START + TRAINING_WEEKS  # relative week 9

# ---------------------------------------------------------------------------
# State configurations
# ---------------------------------------------------------------------------
AP_CONFIG = {
    'name': 'AP',
    'install_schedule': [5] * 8,           # 40 villages over 8 weeks at constant rate
    'cal_week_offset': 0,                  # starts at calendar week 1
    'n_sites': 40,
    'n_treated': 20,
}


def make_install_weeks(state_config):
    """Returns array of installation calendar weeks for a state."""
    weeks = []
    for week_idx, n_villages in enumerate(state_config['install_schedule']):
        cal_week = state_config['cal_week_offset'] + week_idx + 1
        weeks.extend([cal_week] * n_villages)
    return np.array(weeks)


def mean_treatment_weeks(state_config, study_end_week=None):
    """Average treatment duration (weeks) for sites in a state."""
    if study_end_week is None:
        study_end_week = STUDY_END_WEEK
    install_wks = make_install_weeks(state_config)
    treatment_starts = install_wks + STABILIZATION_WEEKS + TRAINING_WEEKS
    durations = study_end_week - treatment_starts
    durations = np.maximum(durations, 0)
    return durations.mean()


# ---------------------------------------------------------------------------
# Load sweep ranges from CSV
# ---------------------------------------------------------------------------
_SWEEP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_params.csv')
_sweep_df = pd.read_csv(_SWEEP_CSV)
_SWEEP = {}
for _, row in _sweep_df.iterrows():
    vals = row['values'].strip()
    if ',' in vals:
        _SWEEP[row['parameter']] = [float(v) for v in vals.split(',')]
    else:
        _SWEEP[row['parameter']] = float(vals)

# For a binary outcome, SD is determined by the mean: sigma = factor * sqrt(mu*(1-mu))
SIGMA_FACTOR = _SWEEP['sigma_factor']


def sigma_from_mu(mu):
    """Derive cross-site heterogeneity SD from baseline compliance rate."""
    return SIGMA_FACTOR * np.sqrt(mu * (1 - mu))

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
# NOTE: target_att is the expected dynamic effect on outcomes (the estimand).
# tau is derived using the finite-horizon AR(1) amplification formula.
PARAM_GRID = {
    'mu_baseline': _SWEEP['mu_baseline'],
    'target_att': _SWEEP['target_att'],
    'rho': _SWEEP['rho'],
    'h_init': _SWEEP['h_init'],
}

STUDY_DURATIONS = {'6 months': 26, '1 year': 52}
