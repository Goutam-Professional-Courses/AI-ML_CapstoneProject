import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import AcquisitionFunctions as af


def populate_subplot(
    subplot: Axes,
    inpt_vs_otpt_pairs: list[dict],
    inpt_vs_otpt_sigma_pairs: list[dict],
    inpt_idx: int,
    confid_intvl: float,
    xticks: np.ndarray,
    yticks: np.ndarray,
    sample_input_points: np.ndarray,
    sample_output_values: np.ndarray,
):
    datapoints: dict = inpt_vs_otpt_pairs[inpt_idx]
    sigmapoints: dict = inpt_vs_otpt_sigma_pairs[inpt_idx]
    axes_keys = list(datapoints)
    sigma_axes_keys = list(sigmapoints)
    otpt_values = datapoints[axes_keys[1]]
    otpt_sigmas = sigmapoints[sigma_axes_keys[1]]
    z_score = af.calc_z_score(confid_intvl)
    lwr_confid_lmt = otpt_values - z_score * otpt_sigmas
    upr_confid_lmt = otpt_values + z_score * otpt_sigmas
    subplot.plot(axes_keys[0], axes_keys[1], "", data=datapoints)
    confid_intvl_lbl = "{}% confidence interval".format(100 * confid_intvl)
    subplot.fill_between(x=sigmapoints[sigma_axes_keys[0]], y1=lwr_confid_lmt, y2=upr_confid_lmt, color="mediumseagreen", alpha=0.4, label=confid_intvl_lbl)
    subplot.legend(loc="lower left", fontsize="medium", facecolor="#ff6666", labelcolor="white", framealpha=1.0)
    subplot.set_xlabel("Input {}".format(inpt_idx + 1))
    subplot.set_ylabel("Output")
    subplot.set_xticks(xticks)
    subplot.set_yticks(yticks)
    subplot.hlines(yticks, xmin=np.min(xticks), xmax=np.max(xticks), colors=["c", "m", "y"])
    subplot.scatter(x=sample_input_points[:, inpt_idx], y=sample_output_values, c="tab:brown", marker="x")


def plot2D(
    weekNbr: int,
    funcNbr: int,
    grid_input_points: np.ndarray,
    grid_output_values: np.ndarray,
    grid_output_sigmas: np.ndarray,
    confid_intvl: float,
    sample_input_points: np.ndarray,
    sample_output_values: np.ndarray,
    output_lower_limit: float = 0.0,
    output_upper_limit: float = 1.0,
    output_step: float = 0.1,
):
    map_inpt_vs_otpt = lambda inpt_idx: dict(Xaxis=grid_input_points[:, inpt_idx], Yaxis=grid_output_values)
    map_inpt_vs_otpt_sigma = lambda inpt_idx: dict(Xaxis=grid_input_points[:, inpt_idx], Yaxis=grid_output_sigmas)
    grid_input_cnt: int = np.size(grid_input_points, axis=1)
    sample_input_cnt: int = np.size(sample_input_points, axis=1)
    sample_input_dtpts_cnt: int = np.size(sample_input_points, axis=0)
    sample_output_values_cnt: int = np.size(sample_output_values)

    inpt_vs_otpt_pairs: list[dict] = list(map(map_inpt_vs_otpt, range(grid_input_cnt)))
    inpt_vs_otpt_sigma_pairs: list[dict] = list(map(map_inpt_vs_otpt_sigma, range(grid_input_cnt)))

    fig = plt.figure(figsize=(14, 10))
    print(
        f"Week: {weekNbr}, function: {funcNbr}, plotting for {sample_input_cnt} input features, each having {sample_input_dtpts_cnt} data-points and {sample_output_values_cnt} output values."
    )
    nrows: int = 1 if grid_input_cnt <= 4 else 2
    ncols: int = min(grid_input_cnt, 4)
    xticks = np.arange(0, 1.1, step=0.1)
    yticks = np.arange(output_lower_limit, output_upper_limit + output_step, step=output_step)

    for inpt_idx in range(grid_input_cnt):
        subplot = fig.add_subplot(nrows, ncols, inpt_idx + 1)
        populate_subplot(
            subplot,
            inpt_vs_otpt_pairs,
            inpt_vs_otpt_sigma_pairs,
            inpt_idx,
            confid_intvl,
            xticks,
            yticks,
            sample_input_points,
            sample_output_values,
        )

    fig_title = "Week: {}, Function: {}".format(weekNbr, funcNbr)
    fig.suptitle(
        fig_title,
        horizontalalignment="center",
        verticalalignment="top",
        fontsize="large",
        fontweight="bold",
    )
    plt.tight_layout()


def print_min_max_output(caption: str, X_inputs: np.ndarray, Y_outputs: np.ndarray):
    input_cnt: int = np.size(X_inputs, axis=1)
    Y_min_idx = np.argmin(Y_outputs)
    Y_max_idx = np.argmax(Y_outputs)
    Y_min = Y_outputs[Y_min_idx]
    Y_max = Y_outputs[Y_max_idx]

    print(caption)
    print("--------------------------------------------------------------------------------------------------------")
    for inpt_idx in range(input_cnt):
        X_inpt_points = X_inputs[:, inpt_idx]
        X_at_Y_min = X_inpt_points[Y_min_idx]
        X_at_Y_max = X_inpt_points[Y_max_idx]
        print(f"Minimum output = {Y_min} when input {inpt_idx + 1} = {X_at_Y_min}")
        print(f"Maximum output = {Y_max} when input {inpt_idx + 1} = {X_at_Y_max}")

    print("")
