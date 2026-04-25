from pathlib import Path
import sys

sys.path.append("../")
import DataLoader as dldr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel as C
import Graphs as grph

# Set root directory to load data-points, week & function numbers.
rootDir: Path = Path("..")
weekNbr: int = 1
funcNbr: int = 1

X_inputs = dldr.load_cumulative_inputs(rootDir, weekNbr, funcNbr)
Y_outputs = dldr.load_cumulative_outputs(rootDir, weekNbr, funcNbr)

# Print minimum & maximum output values (actual) so far and corresponding input co-ordinates.
caption = "Minimum & maximum output values (actual) so far and corresponding input co-ordinates."
grph.print_min_max_output(caption, X_inputs, Y_outputs)

# -----------------------------
# Define Gaussian Process model
# -----------------------------
# Kernel: RBF
kernel: Kernel = RBF(length_scale=1.0, length_scale_bounds="fixed")
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit the model
model.fit(X_inputs, Y_outputs)

# -----------------------------
# Create a grid for visualization
# -----------------------------
grid_size = 500
x_grid = np.linspace(0, 1, 2 * grid_size).reshape(-1, 2)

# Apply the model against points on an evaluation grid and capture the predicted mean & standard deviation.
y_pred_means, y_pred_sigmas = model.predict(x_grid, return_std=True)

# Print minimum & maximum output values (actual + predicted) so far and corresponding input co-ordinates.
caption = "Minimum & maximum output values (actual + predicted) so far and corresponding input co-ordinates."
grph.print_min_max_output(caption, x_grid, y_pred_means)

# -----------------------------
# Plot mean prediction surface
# -----------------------------
confid_intvl: float = 0.80
grph.plot2D(weekNbr, funcNbr, x_grid, y_pred_means, y_pred_sigmas, confid_intvl, X_inputs, Y_outputs, output_upper_limit=0.1, output_step=0.01)
plt.show()
