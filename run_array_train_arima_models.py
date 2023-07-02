# SPDX-License-Identifier: BSD-3-Clause

"""
This script trains a set of ARIMA models for forecasting and backcasting using different parameters
for the AutoRegressive (AR), Integrated (I), and Moving Average (MA) components. Only the best model
(lowest BIC) is saved.

This script is specifically for the 2021 ERCOT backcasted load profiles. Unfortunately, all load
profile classes have missing values from `2021-03-14 23:00:00` to `2021-03-14 23:45:00`
(indices=[7004, 7005, 7006, 7007]). The forecasting and backcasting ARIMA models that are generated
from this script should be used in an ensemble to impute these missing values.

An array job index from some HPC tool is used to train ARIMA models for a given set of parameters
for a specific time series, based on the array job index.

Note
----
Model training is seperated out from the profile generation so that models can be trained in a
distributed fashion (i.e. array job).
"""

import itertools
import json
import os
import sys

from typing import Sequence, Tuple
from nptyping import Float, NDArray, Shape

from rich.progress import track

import numpy as np
import pyarrow.feather
import statsmodels.tsa.arima.model

import research

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.

    Arguments
    ---------
    array_index : int
        The array job ID provided by the HPC tool. It is assumed that the index starts at 1. This
        index will correspond to a specific ERCOT load profile class (there a 248 classes).

    Examples
    --------
    The following will find the best forecast and backcast ARIMA model for the BUSHIDG_COAST ERCOT
    load profile class.

    python3 run_train_arima_models.py 1
    """
    #***********************************************************************************************
    # Get command line arguemnts.
    #***********************************************************************************************
    array_index = int(sys.argv[1])

    if (array_index < 1) or (array_index > 248):
        raise ValueError(f"{array_index} is an invalid index. It must be in the range [1, 248].")

    profile_index = array_index - 1

    #***********************************************************************************************
    # Load data and make parameters.
    #***********************************************************************************************
    basepath = os.getcwd()

    raw_profiles_df = pyarrow.feather.read_feather(
        f"{basepath}/{research.parameters.filepath_profiles_raw}"
    )

    profile_classes = raw_profiles_df.columns.tolist()[1:]
    profile_class = profile_classes[profile_index]

    series = raw_profiles_df[profile_class].to_numpy()

    range_ar = list(range(0, 5))
    range_i = list(range(0, 3))
    range_ma = list(range(0, 5))

    parameters = list(itertools.product(range_ar, range_i, range_ma))

    print(len(parameters))

    #***********************************************************************************************
    # Train models and only keep the models with the lowest BIC.
    #***********************************************************************************************
    nan_indices = np.where(np.isnan(series))[0]
    nan_start = nan_indices[0]
    nan_end = nan_indices[-1]

    series_fore = series[:nan_start]
    series_back = np.flip(series[nan_end + 1:])

    (model_coefficients_fore, orders_fore) = _make_best_arima_result(
        parameters=parameters, series=series_fore
    )

    (model_coefficients_back, orders_back) = _make_best_arima_result(
        parameters=parameters, series=series_back
    )

    #***********************************************************************************************
    # Save the best models.
    #
    # We save the models using a custom JSON schema instead of the pickle version of the models.
    # The pickle version requires ~52 GB of disk space for all models whereas the JSON version only
    # requires 2 MB.
    #***********************************************************************************************
    basepath = os.getcwd()

    filepath_fore = f"{basepath}/models/model-{profile_class}-forecast.json"
    filepath_back = f"{basepath}/models/model-{profile_class}-backcast.json"

    with open(file=filepath_fore, mode="w", encoding="utf-8") as handle:
        json_dict = {
            "profile_class": profile_class,
            "orders": dict(zip(["ar", "diff", "ma"], orders_fore)),
            "parameters": model_coefficients_fore
        }

        json.dump(obj=json_dict, fp=handle, indent=4)

    with open(file=filepath_back, mode="w", encoding="utf-8") as handle:
        json_dict = {
            "profile_class": profile_class,
            "orders": dict(zip(["ar", "diff", "ma"], orders_back)),
            "parameters": model_coefficients_back
        }

        json.dump(obj=json_dict, fp=handle, indent=4)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _make_best_arima_result(
    parameters: Sequence[Tuple[int, int, int]], series: NDArray[Shape["*"], Float]
) -> Tuple[NDArray[Shape["*"], Float], statsmodels.tsa.arima.model.ARIMAResults]:
    """
    This functions takes a list of ARIMA `parameters` and saves the model with the lowest Baysian
    Information Criterion (BIC) to disk.

    Parameters
    ----------
    parameters : list of tuple of (int, int, int)
        The list of all ARIMA parameters to use.
    series : numpy.ndarray of float, (n_timestep,)
        The series data to train on.

    Returns
    -------
    best_model_coefficients : numpy.ndarray of float, (n_coefficient,)
        The estimated AR/I/MA coefficients and error.
    best_orders : list of int
        The AR/I/MA orders that gave the lowest BIC.
    """
    best_bic = np.inf
    best_model_coefficients = None
    best_orders = None

    for (order_ar, order_i, order_ma) in track(parameters, "Training models..."):
        try:
            model = statsmodels.tsa.arima.model.ARIMA(
                endog=series, order=(order_ar, order_i, order_ma)
            )

            result = model.fit()
            model_coefficients = model.fit(return_params=True)

        except Exception: # pylint: disable=broad-except
            # Some set of parameters can result in numerical errors as well as non-converged models.
            # For these cases, we want to simply ignore the error and move on to the next set of
            # parameters.
            continue
        else:
            if result.bic < best_bic:
                best_bic = result.bic
                best_model_coefficients = dict(zip(model.param_names, model_coefficients))
                best_orders = [order_ar, order_i, order_ma]

    return (best_model_coefficients, best_orders)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
