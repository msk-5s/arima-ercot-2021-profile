#!/bin/bash
# <Place HPC tool specific headers here>

# SPDX-License-Identifier: BSD-3-Clause

# This bash script is for submitting an array job to a High Performance Computing (HPC) tool such
# as SLURM. Depending on the tool being used, you may only need to change `SLURM_ARRAY_TASK_ID` in
# the last section to the environment variable that is approriate for your HPC tool. Prepend any
# tool specific headers at line 2 above. 
# The array index should be in the range [1, 248].

# Insert commands here.
#python3 run_array_train_arima_models.py $SLURM_ARRAY_TASK_ID
