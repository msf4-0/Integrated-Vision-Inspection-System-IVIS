"""
Title: Helper
Date: 19/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from typing import Union, List, Dict
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger


def split_string(string):

    # Split the string based on space delimiter
    list_string = string.split(' ')

    return list_string


def join_string(list_string):

    # Join the string based on '-' delimiter
    string = '-'.join(list_string)

    return string


def is_empty(iterable: Union[List, Dict, set]) -> bool:
    return not bool(iterable)


def create_dataframe(data: Union[List, Dict, pd.Series], column_names: List = None, sort_by_ID: bool = False, date_time_format: bool = False) -> pd.DataFrame:
    if data:

        df = pd.DataFrame(data, columns=column_names)
        df.index.name = ('No.')
        if date_time_format:
            df['Date/Time'] = pd.to_datetime(df['Date/Time'],
                                             format='%Y-%m-%d %H:%M:%S')

            df.sort_values(by=['Date/Time'], inplace=True,
                           ascending=False, ignore_index=True)
        elif sort_by_ID:

            df.sort_values(by=['ID'], inplace=True,
                           ascending=True, ignore_index=True)

            # dfStyler = df.style.set_properties(**{'text-align': 'center'})
            # dfStyler.set_table_styles(
            #     [dict(selector='th', props=[('text-align', 'center')])])

        return df
