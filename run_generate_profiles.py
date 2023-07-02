# SPDX-License-Identifier: BSD-3-Clause

"""
This script downloads the ERCOT backcasted loadprofiles and tranforms them from MS Excel format into
a more programatically friendly format.

Note
----
The final imputed load profiles can't be generated unless the ARIMA models are trained beforehand.
See `run_array_train_arima_models.py`.

From a clean start, run this script to download the ERCOT profiles and generate the raw load
profiles. An error message will then ask you to train the ARIMA models. It is highly recommended to
use HPC resources to train the models. After the models are trained, then this script can be ran
again to produce the final load profiles with imputed values.
"""

import json
import os
import zipfile

from rich.progress import track

import numpy as np
import pandas as pd
import pyarrow.feather
import statsmodels.tsa.arima.model
import wget

import research

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    # Get the path to the current folder.
    basepath = os.getcwd()

    folderpath_cache = f"{basepath}/cache"

    #***********************************************************************************************
    # Download the raw load profiles.
    #***********************************************************************************************
    filepath_sheet_zip = f"{basepath}/{research.parameters.filepath_sheet_zip}"
    filepath_sheet = f"{basepath}/{research.parameters.filepath_sheet}"

    if not os.path.exists(filepath_sheet_zip):
        print(f"'{filepath_sheet_zip}' doesn't exists. Downloading...")
        wget.download(research.parameters.url, folderpath_cache)

    if not os.path.exists(filepath_sheet):
        with zipfile.ZipFile(filepath_sheet_zip, "r") as handle:
            handle.extractall(folderpath_cache)

    #***********************************************************************************************
    # Load the raw load profiles.
    #***********************************************************************************************
    raw_profile_df = _generate_raw_profiles(basepath=basepath)

    #***********************************************************************************************
    # Impute missing values using pretrained models.
    #***********************************************************************************************
    filepath_final_profiles = f"{basepath}/{research.parameters.filepath_profiles}"

    final_profile_df = _make_final_profiles(basepath=basepath, raw_profile_df=raw_profile_df)

    pyarrow.feather.write_feather(df=final_profile_df, dest=filepath_final_profiles)

    print("*"*50)
    print("Done!")
    print("*"*50)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _generate_raw_profiles(basepath: str) -> None:
    # pylint: disable=too-many-locals
    """
    Generate the raw ERCOT load profile data. If the raw data hasn't been made yet, then it will be
    made (this will be a fairly long process).

    Parameters
    ----------
    basepath : str
        The base path of the current folder.

    Returns
    -------
    pandas.DataFrame
        The raw ERCOT load profiles.
    """
    year = research.parameters.year
    filepath_profiles_raw = f"{basepath}/{research.parameters.filepath_profiles_raw}"
    filepath_sheet = f"{basepath}/{research.parameters.filepath_sheet}"

    if os.path.exists(filepath_profiles_raw):
        return pyarrow.feather.read_feather(source=filepath_profiles_raw)

    print(f"'{filepath_profiles_raw}' does not exist. Generating...")

    #***********************************************************************************************
    # Load all the month sheets from the spreadsheet into memory.
    #***********************************************************************************************
    months = [
        "January", "February", "March", "April", "May", "June", "July" ,"August", "SEPTEMBER",
        "OCTOBER", "NOVEMBER", "DECEMBER"
    ]

    column_names = ["int_kWh" + str(i) for i in range(1, 97)]

    month_dfs = []

    # Since reading MS Excel spreadsheet data is slow (at least via pandas), we cache the results on
    # disk. This is in case debugging needs to be done later on.
    for month in track(months, "Reading/caching monthly sheet data..."):
        filepath_month = f"{basepath}/cache/sheet-{year}-{month}.feather"
        month_df = None

        if os.path.exists(filepath_month):
            month_df = pyarrow.feather.read_feather(source=filepath_month)
        else:
            # Kinda rediculous how long this takes...
            # 'PType_WZ' is the name of the first column in the excel file (for the load classes and
            # their weather zone).
            month_df = pd.read_excel(
                io=filepath_sheet, sheet_name=month
            )[["PType_WZ"] + column_names]

            pyarrow.feather.write_feather(df=month_df, dest=filepath_month)

        month_dfs.append(month_df)

    #***********************************************************************************************
    # Extract data from the spread sheet.
    #***********************************************************************************************
    # Get the first sheet to determine the unique load profile classes.
    profile_classes = month_dfs[0]["PType_WZ"].unique()

    # Since the load profiles are sampled at 15-minutes intervals over a year, this gives 35040
    # timesteps.
    raw_profiles = np.zeros(shape=(35040, len(profile_classes)), dtype=float)

    timesteps_per_day = 96

    for (col, profile_class) in track(
        enumerate(profile_classes), "Generating load profiles...", total=len(profile_classes)
    ):
        time_start = 0

        for month_df in month_dfs:
            profile_df = month_df.loc[month_df["PType_WZ"] == profile_class, column_names]

            day_count = profile_df.shape[0]
            time_end = time_start + (day_count * timesteps_per_day)

            # Reshape to aggregate the data from multiple day vectors to a single month vector.
            raw_profiles[time_start:time_end, col] = profile_df.to_numpy(dtype=float).reshape(-1,)

            time_start = time_end

    #***********************************************************************************************
    # Add datetime stamps and create a data frame.
    #***********************************************************************************************
    datetime_df = pd.DataFrame(data={
        "datetime": pd.date_range(
            start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:45:00", freq="15min"
            )
    })

    raw_profile_df = pd.concat(
        objs=[datetime_df, pd.DataFrame(data=raw_profiles, columns=profile_classes)],
        axis=1
    )

    pyarrow.feather.write_feather(df=raw_profile_df, dest=filepath_profiles_raw)

    return raw_profile_df

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _make_final_profiles(basepath: str, raw_profile_df: pd.DataFrame) -> pd.DataFrame:
    # pylint: disable=too-many-locals
    """
    Make the final ERCOT load profiles with the missing values imputed.

    The 2021 ERCOT backcasted load profiles all have missing values from `2021-03-14 23:00:00` to
    `2021-03-14 23:45:00` (four timesteps). A simple ensemble composed of an ARIMA forecast and
    backcast model are used to impute the missing values for each load profile.

    Parameters
    ----------
    basepath : str
        The path to the current working directory.
    raw_profile_df : pandas.DataFrame
        The raw ERCOT load profiles with missing values.

    Returns
    -------
    pandas.DataFrame
        The ERCOT load profiles with the missing values imputed using the pretrained models.
    """
    final_profile_df = raw_profile_df.copy(deep=True)
    profile_classes = final_profile_df.columns[1:]

    #***********************************************************************************************
    # Impute the missing values for each profile class.
    #***********************************************************************************************
    for (i, profile_class) in track(
        enumerate(profile_classes), "Imputing missing values...", total=len(profile_classes)
    ):
        filepath_fore = f"{basepath}/models/model-{profile_class}-forecast.json"
        filepath_back = f"{basepath}/models/model-{profile_class}-backcast.json"

        if (not os.path.exists(filepath_fore)) or (not os.path.exists(filepath_back)):
            raise ValueError(" ".join([
                f"Models for {profile_class} are missing.",
                f"Run `python3 run_train_arima_models.py {i + 1}` to create the models for " +
                f"{profile_class}."
            ]))

        #*******************************************************************************************
        # Load JSON results.
        #*******************************************************************************************
        with open(file=filepath_fore, mode="r", encoding="utf-8") as handle:
            json_fore = json.load(fp=handle)

        with open(file=filepath_back, mode="r", encoding="utf-8") as handle:
            json_back = json.load(fp=handle)

        #*******************************************************************************************
        # Get the data for forecasting/backcasting.
        #*******************************************************************************************
        series = final_profile_df.loc[:, profile_class].to_numpy().reshape(-1,)

        nan_indices = np.where(np.isnan(series))[0]
        nan_count = len(nan_indices)
        nan_start = nan_indices[0]
        nan_end = nan_indices[-1]

        series_fore = series[:nan_start]
        series_back = np.flip(series[nan_end + 1:])

        #*******************************************************************************************
        # Load the pretrained ARIMA models.
        #*******************************************************************************************
        model_fore = statsmodels.tsa.arima.model.ARIMA(
            endog=series_fore, order=tuple(json_fore["orders"].values())
        )

        model_back = statsmodels.tsa.arima.model.ARIMA(
            endog=series_back, order=tuple(json_back["orders"].values())
        )

        with model_fore.fix_params(json_fore["parameters"]):
            result_fore = model_fore.fit()

        with model_back.fix_params(json_back["parameters"]):
            result_back = model_back.fit()

        #*******************************************************************************************
        # Imputation.
        #*******************************************************************************************
        predictions_fore = result_fore.forecast(steps=nan_count)
        predictions_back = result_back.forecast(steps=nan_count)
        predictions_ensemble = (predictions_fore + np.flip(predictions_back)) / 2

        final_profile_df.loc[nan_indices, profile_class] = predictions_ensemble

    return final_profile_df

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
