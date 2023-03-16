# import libraries
import pandas as pd

from ML_for_WDN.preprocess_utils import (
    prepare_time_series,
)

PREPARE_NO_LEAK_DATA = True
PREPARE_LEAK_DATA = True

# Sensors to work with
SENSORS = [1, 2, 3, 6, 8, 9, 11, 13]
SAMPLING_PERIOD = .3

def main():

    ##### Prepare no leak data #####
    if PREPARE_NO_LEAK_DATA:
        df_no_leak = pd.read_excel(
            'data/raw_data/no_leak.xlsx', 
            skiprows=0,
            sheet_name=None,
            
            header=0,
        )
        df_no_leak = prepare_time_series(
            df=df_no_leak, 
            sensors=SENSORS,
            sampling_period=SAMPLING_PERIOD,
            )
        df_no_leak.to_csv('data/processed_data/no_leak.csv', index=False)

    ##### Prepare leak data #####
    if PREPARE_LEAK_DATA:
        df_leak = pd.read_excel(
            'data/raw_data/leak_loc_1.xlsx', 
            skiprows=0,
            sheet_name=None,
            header=0,
        )
        df_leak = prepare_time_series(
            df=df_leak, 
            sensors=SENSORS,
            sampling_period=SAMPLING_PERIOD,
            )
        df_leak.to_csv('data/processed_data/leak_loc_1.csv', index=False)


if __name__ == '__main__':
    main()