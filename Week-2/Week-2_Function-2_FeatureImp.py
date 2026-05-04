from pathlib import Path
import sys

from sklearn.gaussian_process import GaussianProcessRegressor

sys.path.append("../")
import DataLoader as dldr
import ModelTrainer as mtrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Kernel, RBF
from sklearn.inspection import permutation_importance

import Graphs as grph

# Set root directory to load data-points, week & function numbers.
rootDir: Path = Path("..")
weekNbr: int = 2
funcNbr: int = 2

X_inputs = dldr.load_cumulative_inputs(rootDir, weekNbr, funcNbr)
Y_outputs = dldr.load_cumulative_outputs(rootDir, weekNbr, funcNbr)

kernel: Kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
grid_size = 20

# Apply the model against points on an evaluation grid and capture the predicted mean & standard deviation.
x_grid, y_pred_means, y_pred_covs = mtrn.runGPR(X_inputs, Y_outputs, 2, model, grid_size)
y_pred_sigmas = np.sqrt(np.diag(y_pred_covs))

# -----------------------------
# Calculate importance of individual features.
# -----------------------------
feature_importances: dict = permutation_importance(model, x_grid, y_pred_means, n_repeats=30, random_state=0)
grph.plotFeatureImportance(weekNbr, funcNbr, feature_importances["importances_mean"], feature_importances["importances_std"])
plt.show()
