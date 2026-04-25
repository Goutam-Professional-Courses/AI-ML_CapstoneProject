import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel
from sklearn.model_selection import cross_val_score
import argparse
from pathlib import Path
import sys

sys.path.append("./")
import DataLoader as dldr


def select_best_gpr_kernel(X, y, cv=5, scoring="neg_mean_squared_error"):
    """
    Automatically tests multiple Gaussian Process kernels and selects the best one.

    Parameters:
        X (array-like): Feature matrix of shape (n_samples, n_features)
        y (array-like): Target vector of shape (n_samples,)
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric for cross_val_score

    Returns:
        dict: Best kernel, its mean score, and all tested results
    """
    # Define candidate kernels
    candidate_kernels = [
        RBF(length_scale=1.0, length_scale_bounds='fixed'),
        Matern(length_scale=1.0, length_scale_bounds='fixed'),
        RationalQuadratic(length_scale=1.0, length_scale_bounds='fixed'),
        RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1.0),
        RationalQuadratic(length_scale=1.0, alpha=0.1) + WhiteKernel(noise_level=1.0),
        ExpSineSquared(length_scale=1.0, periodicity=3.0) + WhiteKernel(noise_level=1.0),
        (RBF(length_scale=1.0) + DotProduct()) + WhiteKernel(noise_level=1.0),
        (Matern(length_scale=1.0, nu=0.5) * ExpSineSquared(length_scale=1.0, periodicity=3.0)) + WhiteKernel(noise_level=1.0),
    ]

    results = []
    best_score = -np.inf
    best_kernel = None

    for kernel in candidate_kernels:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
        try:
            scores = cross_val_score(gpr, X, y, cv=cv, scoring=scoring)
            mean_score = scores.mean()
            results.append((kernel, mean_score))
            print(f"Kernel: {kernel}\n  Mean CV Score: {mean_score:.6f}\n")
            if mean_score > best_score:
                best_score = mean_score
                best_kernel = kernel
        except Exception as e:
            print(f"Skipping kernel {kernel} due to error: {e}")

    return {"best_kernel": best_kernel, "best_score": best_score, "all_results": results}


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read week and function numbers from the command line.")
    parser.add_argument("weekNbr", type=int)
    parser.add_argument("funcNbr", type=int)
    args = parser.parse_args()
    print(f"Select the most suitable kernel for week: {args.weekNbr}, function: {args.funcNbr}.")

    rootDir: Path = Path(".")
    weekNbr: int = args.weekNbr
    funcNbr: int = args.funcNbr

    X_inputs = dldr.load_cumulative_inputs(rootDir, weekNbr, funcNbr)
    Y_outputs = dldr.load_cumulative_outputs(rootDir, weekNbr, funcNbr)

    print(f"X_inputs.shape: {X_inputs.shape}, Y_outputs.shape: {Y_outputs.shape}")

    result = select_best_gpr_kernel(X_inputs, Y_outputs, cv=5)
    print("\nBest Kernel:", result["best_kernel"])
    print("Best CV Score:", result["best_score"])
