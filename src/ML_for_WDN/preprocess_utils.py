import pandas as pd
import traces
from datetime import timedelta
#import sdt.changepoint as sdt

def prepare_time_series(df, sensors, sampling_period=.5):
    for key in df.keys():
        df[key]['Time'] = df[key]['Time'].apply(lambda x: x.strftime('%H:%M:%S'))

        # Remove duplicates in Time column
        df[key] = df[key].loc[df[key]['Time'].shift() != df[key]['Time']]

        # Combine Date and Time columns to datetime
        df[key] = df[key].loc[(df[key]['Time'] >= '00:30:00') & (df[key]['Time'] <= '23:30:00')]

        df[key]['Time'] = pd.to_datetime(
            df[key]['Date'].astype(str) + ' ' + df[key]['Time'].astype(str)
        )

        df[key] = interpolate_time_series(df[key], sensors)

    df_sensors_no_leak = pd.concat(df, ignore_index=True)

    return df_sensors_no_leak


def interpolate_time_series(df, sensors, sampling_period=.5):

    df_sensors = []
    for i in sensors:
        if i < 10:
            sensor_name = f'FM0{i}'
        else:
            sensor_name = f'FM{i}'
            
        df_sensor_i_flow = df[['Time', f'{sensor_name}_flow']]
        df_sensor_i_pressure = df[['Time', f'{sensor_name}_head']]

        ts_flow = traces.TimeSeries([
            (df_sensor_i_flow['Time'].iloc()[j], df_sensor_i_flow[f'{sensor_name}_flow'].iloc()[j]) 
            for j in range(len(df_sensor_i_flow))
        ])
        ts_pressure = traces.TimeSeries([
            (df_sensor_i_pressure['Time'].iloc()[j], df_sensor_i_pressure[f'{sensor_name}_head'].iloc()[j]) 
            for j in range(len(df_sensor_i_pressure))
        ])
        interpolated_timeseries_flow = ts_flow.sample(
            sampling_period=timedelta(minutes=sampling_period),
            start=df['Time'].iloc()[0],
            end=df['Time'].iloc()[-1],
            interpolate='linear',
        )
        interpolated_timeseries_pressure = ts_pressure.sample(
            sampling_period=timedelta(minutes=sampling_period),
            start=df['Time'].iloc()[0],
            end=df['Time'].iloc()[-1],
            interpolate='linear',
        )
        interpolated_df_flow = pd.DataFrame(
            interpolated_timeseries_flow, 
            columns=['Time', f'flow_{i}']
        )
        interpolated_df_pressure = pd.DataFrame(
            interpolated_timeseries_pressure, 
            columns=['Time', f'pressure_{i}']
        )
        df_sensor_i = pd.merge(
            interpolated_df_flow,
            interpolated_df_pressure,
            on='Time',
        )

        df_sensors.append(df_sensor_i)

    df_sensors = pd.concat(df_sensors, axis=1)

    df_sensors = df_sensors.loc[:,~df_sensors.columns.duplicated()].copy()

    return df_sensors