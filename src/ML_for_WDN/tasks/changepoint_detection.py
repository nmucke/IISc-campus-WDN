from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pdb
import sdt.changepoint as sdt
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)


DATA_PATH = 'data/processed_data/leak_loc_1.csv'
#DATA_PATH = 'data/processed_data/no_leak.csv'

COLUMNS = [
    'flow_1', 'flow_2', 'flow_3', 'flow_6', 'flow_8', 'flow_9', 'flow_11', 'flow_13',
    'pressure_1', 'pressure_2', 'pressure_3', 'pressure_6', 'pressure_8', 'pressure_9', 'pressure_11', 'pressure_13'
    ]

METHOD = 'pelt'

def moving_average(a, n=3):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():

    # Load data
    data = pd.read_csv(DATA_PATH)
    data['Date'] = data['Time'].apply(lambda x: x.split(' ')[0])
    data['Time'] = data['Time'].apply(lambda x: x.split(' ')[1])
    
    # Get data from just one day
    #data = data[data['Date'] == '2022-11-29']
    data = data[data['Date'] == '2022-10-18']

    # Select columns
    data = data[COLUMNS]

    # Scale data
    #scaler = MinMaxScaler()
    #data = scaler.fit_transform(data)
    
    # To numpy
    time_series = data.to_numpy()
    #time_series = time_series[0:2000]

    # Moving average
    time_series_smooth = moving_average(time_series, n=10)

    # Downsample data
    time_series_smooth = time_series_smooth[::2]

    # Compute difference
    time_series_diff = np.diff(time_series_smooth, axis=0)

    if METHOD == 'bayes_offline':
        det = sdt.BayesOffline(
            prior="geometric", 
            obs_likelihood="full_cov",
            prior_params={'p': 0.9},
            numba_logsumexp=True
        )
        change_points = det.find_changepoints(
            data=time_series_diff, 
            prob_threshold=0.01
        )

    elif METHOD == 'bayes_online':
        det = sdt.BayesOnline(
            hazard="const", 
            obs_likelihood='student_t'
        )
        change_points = det.find_changepoints(
            data=time_series_diff, 
            past=50, 
            prob_threshold=0.1
        )

    elif METHOD == 'pelt':
        det = sdt.Pelt(cost="l1", min_size=5, jump=1)
        change_points = det.find_changepoints(
            data=time_series_diff, 
            penalty=100, 
            #max_exp_cp=4
        )

    #det = sdt.Pelt(cost="l1", min_size=1, jump=10)
    #change_points = det.find_changepoints(time_series, 1000, max_exp_cp=4)


    plt.figure()
    #plt.plot(time_series[:, 0], linewidth=2)
    #plt.plot(time_series[:, 1], linewidth=2)
    plt.plot(time_series_smooth[:, 0], linewidth=2)
    #plt.plot(time_series_smooth[:, 1], linewidth=2)
    #plt.plot(time_series_diff[:, 10], linewidth=2)
    #plt.plot(time_series_diff[:, 11], linewidth=2)
    for i in range(len(change_points)):
        plt.axvline(x=change_points[i], color='r', linewidth=2)
    plt.show()


if __name__ == '__main__':
    main()