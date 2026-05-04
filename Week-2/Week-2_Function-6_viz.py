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
weekNbr: int = 2
funcNbr: int = 6

X_inputs = dldr.load_cumulative_inputs(rootDir, weekNbr, funcNbr)
Y_outputs = dldr.load_cumulative_outputs(rootDir, weekNbr, funcNbr)

kernel: Kernel = Matern(length_scale=1.0, length_scale_bounds="fixed", nu=np.inf)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
grid_size = 500

# Apply the model against points on an evaluation grid and capture the predicted mean & standard deviation.
x_grid, y_pred_means, y_pred_covs = mtrn.runGPR(X_inputs, Y_outputs, 5, model, grid_size)
y_pred_sigmas = np.sqrt(np.diag(y_pred_covs))

# -----------------------------
# Plot mean prediction surface
# -----------------------------
confid_intvl: float = 0.95
grph.plotFunction(
    weekNbr, funcNbr, x_grid, y_pred_means, y_pred_sigmas, confid_intvl, X_inputs, Y_outputs, output_lower_limit=-3.0, output_upper_limit=0, output_step=0.2
)
plt.show()
