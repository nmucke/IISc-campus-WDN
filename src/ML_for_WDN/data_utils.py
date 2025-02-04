import pdb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import wntr
from functools import lru_cache

@lru_cache(maxsize=None)
def load_data(data_path: str, fraction: float = 1.0):

    df = pd.concat(pd.read_excel(data_path, sheet_name=None), ignore_index=True)

    # Convert time column to datetime
    df['Time'] = df['Time'].astype("string")
    df['Time'] = pd.to_datetime(
        df['Time'],
        format='%H:%M:%S'
        )
    df['Time'] = [time.time() for time in df['Time']]

    time_1 = pd.to_datetime('06:00:00', format='%H:%M:%S').time()
    time_2 = pd.to_datetime('11:00:00', format='%H:%M:%S').time()
    time_3 = pd.to_datetime('17:00:00', format='%H:%M:%S').time()
    time_4 = pd.to_datetime('19:00:00', format='%H:%M:%S').time()

    df = df[(df['Time'] >= time_1) & (df['Time'] <= time_2) | (df['Time'] >= time_3) & (df['Time'] <= time_4)]
    
    return df


def clean_dataframes(
    dataframes: list,
    flow_rate_threshold: float = 0.1,
    pressure_threshold: float = 0.1,
    columns_to_use: list = None,
    use_common_columns: bool = True,
    ):

    # Remove columns with NaN values
    for df in dataframes:
        df.dropna(axis=1, inplace=True)

    columns = []
    for df in dataframes:
        columns.append(df.columns.tolist())
    
    # Get the columns that are in all dfs
    common_columns = list(set(columns[0]).intersection(*columns))
    
    # Remove columns that are not to be used
    for df in dataframes:
        for column in df.columns:
            if columns_to_use is not None:
                if column not in columns_to_use:
                    df.drop([column], axis=1, inplace=True)
            elif use_common_columns:
                if column not in common_columns:
                    df.drop([column], axis=1, inplace=True)

    # Remove Date and Time columns
    for df in dataframes:
        for col in ['Date', 'Time']:
            if col in df.columns:
                df.drop([col], axis=1, inplace=True)

    # Set all flow rate between -flow_rate_threshold and flow_rate_threshold to 0
    for df in dataframes:
        for column in df.columns:
            if column[-4:] == 'flow':
                df.loc[df[column].abs() < flow_rate_threshold, column] = 0.0
    
    # Set all pressure between -pressure_threshold and pressure_threshold to 0
    for df in dataframes:
        for column in df.columns:
            if column[-4:] == 'head':
                df.loc[df[column] < pressure_threshold, column] = 0.0
   
    return dataframes
    