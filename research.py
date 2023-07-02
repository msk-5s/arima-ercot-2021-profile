# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains the parameters used to generate results in the research paper.
"""

import dataclasses

#===================================================================================================
#===================================================================================================
@dataclasses.dataclass
class Parameters:
    """
    Parameters used to generate results in the research paper.
    """
    # pylint: disable=too-many-instance-attributes
    filepath_profiles: str
    filepath_profiles_raw: str
    filepath_sheet: str
    filepath_sheet_zip: str
    url: str
    year: int

#***************************************************************************************************
#***************************************************************************************************
# Changes made to the `parameters` should be propagated to the `make_kwargs_map` function.
parameters = Parameters(
    filepath_profiles="data/ercot-2021-load_profiles.feather",
    filepath_profiles_raw="data/ercot-2021-load_profiles-raw.feather",
    filepath_sheet="cache/ERCOT Backcasted Load Profiles 2021.xlsx",
    filepath_sheet_zip="cache/ERCOT_Backcasted_Load_Profiles_2021.zip",
    url="https://www.ercot.com/files/docs/2021/11/05/ERCOT_Backcasted_Load_Profiles_2021.zip",
    year=2021
)
