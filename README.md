# **ercot-2021-profile**

This repository contains the source code for generating clean (no missing data) ERCOT 2021 load profiles. ERCOT maintains load profiles for 248 different load classes that might have some combination of distributed energy resources (DER) (i.e. solar, wind, etc.) across different weather zones within the state of Texas. The profiles are sampled at 15-minute intervals over the year giving a total of 35040 timesteps per profile (96 timesteps per day). Unfortunately, each profile has an hour (4 timesteps) of missing data from '2021-03-14 23:00:0' to '2021-03-14 23:45:0'. An ensemble of a forecasting and backcasting ARIMA model is used to impute these missing values.

These load profiles are used in the [arima-ercot-2021-opendss](https://github.com/msk-5s/arima-ercot-2021-opendss) repository to generate the [`arima-ercot-2021`](https://www.kaggle.com/datasets/msk5sdata/arima-ercot-2021) data suite.

The generated profiles will be in the [Apache Arrow Feather](https://arrow.apache.org/docs/python/feather.html) format.

## Requirements
    - Python 3.8+ (64-bit)
    - See requirements.txt file for the required python packages.

## Folders
`cache/`
: This folder is used to hold the downloaded raw ERCOT 2021 backcasted load profiles and any intermediary data.

`data/`
: This folder contains the raw and imputed load profiles formated as a pandas dataframe and saved in feather format.

`models/`
: This folder contains the trained forecast/backcast ARIMA models for each load class.

## Running
Run the `run_generate_profiles.py` script to download the raw profiles from [ERCOT](https://www.ercot.com/mktinfo/loadprofile/alp), convert the excel sheets to a pandas dataframe, then use the trained ARIMA models to impute the missing values for each load class. 

The `run_array_train_arima_models.py` script is included for anyone interested in training the ARIMA models from scratch (the `models\` folder contains the pretrained models). The `submit_job.sh` bash script is included for submitting array jobs to some HPC tool.

## Converting to `.csv` (if desired)
Pandas can be used to convert a `.feather` file to a `.csv` file using the below code:
```
import pyarrow.feather

# A pandas dataframe is returned.
# We are assuming that we are in the repository's root directory.
data_df = pyarrow.feather.read_feather("data/ercot-2021-load_profiles.feather")

data_df.to_csv("data/ercot-2021-load_profiles.csv")
```
