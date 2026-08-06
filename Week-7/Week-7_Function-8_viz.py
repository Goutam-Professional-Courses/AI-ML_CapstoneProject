from pathlib import Path
import sys

from sklearn.gaussian_process import GaussianProcessRegressor

sys.path.append("../")
import DataLoader as dldr
import ModelTrainer as mtrn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Kernel, RBF, Matern
import Graphs as grph

# Set root directory to load data-points, week & function numbers.
rootDir: Path = Path("..")
weekNbr: int = 7
funcNbr: int = 8

X_inputs = dldr.load_cumulative_inputs(rootDir, weekNbr, funcNbr)
Y_outputs = dldr.load_cumulative_outputs(rootDir, weekNbr, funcNbr)

kernel: Kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=np.inf)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
grid_size = 500

# Apply the model against points on an evaluation grid and capture the predicted mean & standard deviation.
x_grid, y_pred_means, y_pred_covs = mtrn.runGPR(X_inputs, Y_outputs, 8, model, grid_size)
y_pred_sigmas = np.sqrt(np.diag(y_pred_covs))

# -----------------------------
# Plot mean prediction surface
# -----------------------------
confid_intvl: float = 0.80
grph.plotFunction(
    weekNbr, funcNbr, x_grid, y_pred_means, y_pred_sigmas, confid_intvl, X_inputs, Y_outputs, output_lower_limit=5, output_upper_limit=10, output_step=1
)
plt.show()
