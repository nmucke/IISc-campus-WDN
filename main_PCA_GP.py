from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import wntr
import pdb


from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neural_network import MLPRegressor

import ML_for_WDN





from ML_for_WDN.data_utils import clean_dataframes, load_data

def get_reconstruction_error(
    df: pd.DataFrame,
    pca: PCA,
    ):

    # Compute reconstruction error
    reconstruction_error = df - pca.inverse_transform(pca.transform(df))
    l2_reconstruction_error = np.linalg.norm(reconstruction_error, axis=1)
    l2_reconstruction_error = l2_reconstruction_error / np.linalg.norm(df, axis=1)
    mean_reconstruction_error = np.mean(l2_reconstruction_error)

    return np.abs(reconstruction_error)**2, l2_reconstruction_error, mean_reconstruction_error

def main():

    ###### Read in the data ######
    data_files = [
        'data/data_no_leak.xlsx',
        'data/data_leak_1.xlsx',
        'data/data_leak_2.xlsx',
        'data/data_leak_3.xlsx',
        ]

    dfs = [load_data(data_path=data_file) for data_file in data_files]

    # Clean the dataframes
    dfs = clean_dataframes(
        dataframes=dfs,
        flow_rate_threshold=1.0,
        pressure_threshold=2.0,
        )

    # Create a dictionary of dataframes
    dfs_dict = {
        'no_leak': dfs[0],
        'leak_1': dfs[1],
        'leak_2': dfs[2],
        'leak_3': dfs[3],
    }
       
    leak_pipe = {
        'leak_1': ('J-32', 'J-86'),#, 'P-49'),#'P-2',
        'leak_2': ('J-44', 'J-35'),#), 'P-2'),#'P-49',
        'leak_3': ('J-15', 'J-72'),#, 'P-26'),#'P-26',
    }
    # Load epanet model
    wn = wntr.network.WaterNetworkModel('data/IISc_epanet.inp')
    # Get networkx graph
    G = wn.to_graph()

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    preprocessor = MinMaxScaler()
    preprocessor.fit(dfs_dict['no_leak'])
    
    for key, df in dfs_dict.items():
        dfs_dict[key] = preprocessor.transform(df)

    # Perform PCA on the data
    #pca = PCA(n_components=7)
    pca = KernelPCA(
        n_components=10, kernel="rbf", gamma=1, fit_inverse_transform=True, alpha=0.1
    )
   
    # Fit the PCA model on no leak data
    pca.fit(dfs_dict['no_leak'])

    # Get the reconstruction error
    recon_error = {}
    l2_recon_error = {}
    mean_recon_error = {}
    for key, df in dfs_dict.items():
        recon_error[key], l2_recon_error[key], mean_recon_error[key] = \
            get_reconstruction_error(df, pca)

    # Print the mean reconstruction error
    print('Mean reconstruction error:')
    for key, error in mean_recon_error.items():
        print(f'{key}: {error}')

    threshold = l2_recon_error['no_leak'].mean() + 3 * l2_recon_error['no_leak'].std()

    num_anomalies = {}
    num_total = {}
    for key, error in l2_recon_error.items():
        num_anomalies[key] = np.sum(error > threshold)
        num_total[key] = len(error)


    print('Number of anomalies based on PCA:')
    for key, num in num_anomalies.items():
        print(f'{key}: {num} of {num_total[key]}')

    ############################################

    


    # Plot the reconstruction error
    plt.figure()
    for key, error in l2_recon_error.items():
        plt.plot(error[0:13000], label=key)
    plt.axhline(y=threshold, color='k', linestyle='--', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Reconstruction error')
    plt.legend()
    plt.show()

    square_domain = {
        'x_min': 1e8,
        'x_max': -1e8,
        'y_min': 1e8,
        'y_max': -1e8,
    }

    for node, _pos in pos.items():
        if _pos[0] < square_domain['x_min']:
            square_domain['x_min'] = _pos[0]
        if _pos[0] > square_domain['x_max']:
            square_domain['x_max'] = _pos[0]
        if _pos[1] < square_domain['y_min']:
            square_domain['y_min'] = _pos[1]
        if _pos[1] > square_domain['y_max']:
            square_domain['y_max'] = _pos[1]

    # Get the node positions of sensors
    sensor_nodes = [
        'J-4', 'J-63', 'J-47', 'J-32', 'J-14', 'J-12', 'J-19', 'J-44', 'J-45',
    ]
    sensor_nodes_pos = {}
    for node, _pos in pos.items():
        if node in sensor_nodes:
            sensor_nodes_pos[node] = _pos

    sensor_X = np.zeros((9, 2))
    for i in range(0, 9):
        sensor_X[i, 0] = sensor_nodes_pos[sensor_nodes[i]][0]
        sensor_X[i, 1] = sensor_nodes_pos[sensor_nodes[i]][1]

    X = np.zeros((9, 2))
    for i in range(0, 9):
        X[i, 0] = sensor_nodes_pos[sensor_nodes[i]][0]
        X[i, 1] = sensor_nodes_pos[sensor_nodes[i]][1]
                
    num_time_steps = 100
    num_skip_steps = 10
    for case in ['leak_1', 'leak_2', 'leak_3']:
        Z = np.zeros((100, 100))
        Z_std = np.zeros((100, 100))

        for counter, i in enumerate(range(0, num_time_steps, num_skip_steps)):


            
            Y_head = recon_error[case][i:(i+num_skip_steps), 9:].mean(axis=0)
            Y_flow = recon_error[case][i:(i+num_skip_steps), 0:9].mean(axis=0)


            preprocessor = MinMaxScaler()
            preprocessor.fit_transform(Y_head.reshape(-1, 1))
            preprocessor = MinMaxScaler()
            preprocessor.fit_transform(Y_flow.reshape(-1, 1))

            kernel = 1 * RBF(length_scale=1e2, length_scale_bounds=(1e-1, 1e4))
            gp_head = GaussianProcessRegressor(
                kernel=kernel,
                random_state=0,
                n_restarts_optimizer=5,
            )
            gp_head.fit(X, Y_head)

            gp_flow = GaussianProcessRegressor(
                kernel=kernel,
                random_state=0,
                n_restarts_optimizer=5,
            )
            gp_flow.fit(X, Y_flow)


            x = np.linspace(square_domain['x_min'], square_domain['x_max'], 100)
            y = np.linspace(square_domain['y_min'], square_domain['y_max'], 100)
            X_mesh, Y_mesh = np.meshgrid(x, y)
            Z_i_head = np.zeros((100, 100))
            Z_i_std_head = np.zeros((100, 100))
            Z_i_flow = np.zeros((100, 100))
            Z_i_std_flow = np.zeros((100, 100))
            for i in range(0, 100):
                for j in range(0, 100):
                    Z_i_head[i, j], Z_i_std_head = gp_head.predict(np.array([[X_mesh[i, j], Y_mesh[i, j]]]), return_std=True)
                    Z_i_flow[i, j], Z_i_std_flow = gp_flow.predict(np.array([[X_mesh[i, j], Y_mesh[i, j]]]), return_std=True)
            
            Z += Z_i_head/np.max(Z_i_head) + Z_i_flow/np.max(Z_i_flow)
            Z_std += Z_i_std_head/np.max(Z_i_std_head) + Z_i_std_flow/np.max(Z_i_std_flow)

        Z = Z / counter
        
        # Plot the sensor nodes
        plt.figure(figsize=(20, 20))
        plt.contourf(X_mesh, Y_mesh, Z, cmap='Reds')
        plt.colorbar()
        nx.draw_networkx(
            G,
            pos=pos,
            node_size=10,
            node_color='k',
            edge_color='k',
            width=4.,
            with_labels=False,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=pos,
            edge_labels={leak_pipe[case]: 'X'},
            font_size=16,
            font_color='blue',
        )
        plt.scatter(
            sensor_X[:, 0], 
            sensor_X[:, 1], 
            c='blue',
            s=200, 
        )
        plt.title('IISc Water Network')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.savefig(f'leak_{case}.pdf')
        plt.show()

if __name__ == '__main__':
    main()

