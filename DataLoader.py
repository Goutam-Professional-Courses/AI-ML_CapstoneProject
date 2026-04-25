import re
import ast
from pathlib import Path
import linecache
import numpy as np


def load_initial_inputs(initDataBaseDir: Path, funcNbr: int) -> np.ndarray:
    funcName: str = "function_{}".format(funcNbr)
    filePath: Path = initDataBaseDir.joinpath(funcName, "initial_inputs.npy")
    X_init_inputs = np.load(filePath)
    return X_init_inputs


def load_initial_outputs(initDataBaseDir: Path, funcNbr: int) -> np.ndarray:
    funcName: str = "function_{}".format(funcNbr)
    filePath: Path = initDataBaseDir.joinpath(funcName, "initial_outputs.npy")
    Y_init_outputs = np.load(filePath)
    return Y_init_outputs


# Function that can read a list of numpy arrays from a line of text.
def parse_arrays_from_text(lineOfText: str) -> list[np.ndarray]:
    # Capture the list part inside each array([...])
    # Works for simple 1D lists of numbers like your file.
    pattern = re.compile(r"array\(\s*(\[[^\]]*\])\s*\)")
    arrays = []

    for m in pattern.finditer(lineOfText):
        list_text = m.group(1)  # like "[0.5, 0.5, 0.5]"
        data = ast.literal_eval(list_text)  # safe parse to Python list
        arrays.append(np.array(data, dtype=float))

    return arrays


# Function that can read a list of numpy decimal numbers from a line of text.
def parse_floats_from_text(lineOfText: str) -> np.ndarray:
    # Extract numbers inside np.float64(...)
    values = re.findall(r"np\.float64\(([^)]+)\)", lineOfText)

    if not values:
        raise ValueError("No np.float64 values found in file.")

    floats = [float(v) for v in values]
    return np.array(floats, dtype=np.float64)


def load_inputs(rootDir: Path, weekNbr: int, funcNbr: int) -> np.ndarray:
    wkDirName = "Week-{}".format(weekNbr)
    wkInputFile: Path = rootDir.joinpath(wkDirName, "inputs.txt")
    lineOfText: str = linecache.getline(str(wkInputFile), weekNbr)
    thisWeekThisFunctionInputs: np.ndarray
    if not lineOfText.strip():
        thisWeekThisFunctionInputs = np.empty(0)
    else:
        thisWeekAllFunctionsInputs = parse_arrays_from_text(lineOfText)
        thisWeekThisFunctionInputs = thisWeekAllFunctionsInputs[funcNbr - 1]
    return thisWeekThisFunctionInputs


def load_output(rootDir: Path, weekNbr: int, funcNbr: int) -> np.ndarray:
    wkDirName = "Week-{}".format(weekNbr)
    wkOutputFile: Path = rootDir.joinpath(wkDirName, "outputs.txt")
    lineOfText: str = linecache.getline(str(wkOutputFile), weekNbr)
    thisWeekThisFunctionOutput: np.ndarray
    if not lineOfText.strip():
        thisWeekThisFunctionOutput = np.empty(0)
    else:
        thisWeekAllFunctionsOutputs = parse_floats_from_text(lineOfText)
        thisWeekThisFunctionOutput = np.array(thisWeekAllFunctionsOutputs[funcNbr - 1], dtype=float)
    return thisWeekThisFunctionOutput


def load_cumulative_inputs(rootDir: Path, untilWeekNbr: int, funcNbr: int) -> np.ndarray:
    initDataBaseDir: Path = rootDir.joinpath("initial_data")

    # Load initial input data-points
    X_init_inputs = load_initial_inputs(initDataBaseDir, funcNbr)
    X_cumulative_inputs = X_init_inputs.copy()

    for wkIdx in range(1, untilWeekNbr + 1):
        thisFunctionWeeklyInputs = load_inputs(rootDir, wkIdx, funcNbr)

        if thisFunctionWeeklyInputs.size == 0:
            print(f"No data found for week: {wkIdx}, function: {funcNbr}, exiting further loading of inputs.")
            break

        # Combine initial and weekly input arrays.
        X_cumulative_inputs = np.append(X_cumulative_inputs, np.array([thisFunctionWeeklyInputs]), axis=0)

    print(f"Week: {untilWeekNbr}, function {funcNbr}: Combined initial & weekly samples contain {len(X_cumulative_inputs)} input data-points.")
    return X_cumulative_inputs


def load_cumulative_outputs(rootDir: Path, untilWeekNbr: int, funcNbr: int) -> np.ndarray:
    initDataBaseDir: Path = rootDir.joinpath("initial_data")

    # Load initial output values
    Y_init_outputs = load_initial_outputs(initDataBaseDir, funcNbr)
    Y_cumulative_outputs = Y_init_outputs.copy()

    for wkIdx in range(1, untilWeekNbr + 1):
        thisFunctionWeeklyOutput = load_output(rootDir, wkIdx, funcNbr)

        if thisFunctionWeeklyOutput.size == 0:
            print(f"No data found for week: {wkIdx}, function: {funcNbr}, exiting further loading of outputs.")
            break

        # Combine initial and weekly output values.
        Y_cumulative_outputs = np.append(Y_cumulative_outputs, np.array([thisFunctionWeeklyOutput]), axis=0)

    # Compute the maximum output value from the initial + weekly cumulative merged data-set.
    y_curr_max = np.max(Y_cumulative_outputs)
    print(
        f"Week: {untilWeekNbr}, function {funcNbr}: Combined initial & weekly outputs contain {len(Y_cumulative_outputs)} output values, with maximum = {y_curr_max}."
    )
    return Y_cumulative_outputs
