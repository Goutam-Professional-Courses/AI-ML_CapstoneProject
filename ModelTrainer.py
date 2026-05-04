import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import Graphs as grph


def runGPR(
    X_inputs: np.ndarray, Y_outputs: np.ndarray, input_features_count: int, model: GaussianProcessRegressor, grid_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Print minimum & maximum output values (actual) so far and corresponding input co-ordinates.
    caption = "Minimum & maximum output values (actual) so far and corresponding input co-ordinates."
    grph.print_min_max_output(caption, X_inputs, Y_outputs)

    # Fit the model
    model.fit(X_inputs, Y_outputs)

    # -----------------------------
    # Create a grid for visualization
    # -----------------------------
    x_grid = np.linspace(0, 1, input_features_count * grid_length).reshape(-1, input_features_count)

    # Apply the model against points on an evaluation grid and capture the predicted mean & standard deviation.
    y_pred_means, y_pred_covs = model.predict(x_grid, return_cov=True)  # pyright: ignore[reportAssignmentType]

    return (x_grid, y_pred_means, y_pred_covs)
