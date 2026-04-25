import numpy as np
from scipy.stats import norm


def calc_z_score(confid_intvl: float):
    z_score = 0.5 + (confid_intvl / 2)
    z_score = norm.ppf(z_score)
    return z_score


def ucb(confid_intvl: float, y_pred_means: np.ndarray, y_pred_sigmas: np.ndarray) -> np.ndarray:
    z_score = calc_z_score(confid_intvl)
    acquisition_function = y_pred_means + (z_score * y_pred_sigmas)
    return acquisition_function


def prob_improvement(eta: float, y_pred_means: np.ndarray, y_pred_sigmas: np.ndarray, y_curr_max: float) -> np.ndarray:
    z = (y_pred_means - y_curr_max - eta) / (y_pred_sigmas + 1e-12)
    acquisition_function = norm.cdf(z)
    return acquisition_function
