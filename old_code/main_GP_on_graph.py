from logging import warn
import pdb
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import wntr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

#import warn
import tensorflow as tf
#import tensorflow_probability as tfp
from tqdm.notebook import trange
import tqdm
from scipy import sparse

import gpflow

from graph_matern.kernels.graph_matern_kernel import GraphMaternKernel
from graph_matern.kernels.graph_diffusion_kernel import GraphDiffusionKernel

from ML_for_WDN.data_utils import clean_dataframes, load_data

dtype = tf.float64
gpflow.config.set_default_float(dtype)

# Don't use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_eigenpairs = 500
case = 'leak_2'

def optimize_GPR(model, train_steps):
    loss = model.training_loss
    trainable_variables = model.trainable_variables

    adam_opt = tf.optimizers.Adam()
    adam_opt.minimize(loss=loss, var_list=trainable_variables)

    #t = trange(train_steps - 1)
    t = tqdm.tqdm(range(train_steps-1), desc='Training', leave=True)
    for step in t:
        #opt_step(adam_opt, loss, trainable_variables)
        adam_opt.minimize(loss, trainable_variables)
        if step % 200 == 0:
            t.set_postfix({'likelihood': -model.training_loss().numpy()})

@tf.function
def opt_step(opt, loss, variables):
    pdb.set_trace()
    opt.minimize(loss, variables)
    
def get_reconstruction_error(
    df: pd.DataFrame,
    pca: PCA,
    ):

    # Compute reconstruction error
    reconstruction_error = df - pca.inverse_transform(pca.transform(df))
    l2_reconstruction_error = np.linalg.norm(reconstruction_error, axis=1)
    l2_reconstruction_error = l2_reconstruction_error / np.linalg.norm(df, axis=1)
    mean_reconstruction_error = np.mean(l2_reconstruction_error)

    return np.abs(reconstruction_error), l2_reconstruction_error, mean_reconstruction_error

sensor_to_label = {
    'FM01': 'J-4',
    'FM02': 'J-63',
    'FM03': 'J-47',
    'FM04': 'J-33',
    'FM05': 'J-32',
    'FM06': 'J-14',
    'FM07': 'J-77',
    'FM08': 'J-12',
    'FM09': 'J-19',
    'FM10': 'J-58',
    'FM11': 'J-44',
    'FM12': 'J-20',
    'FM13': 'J-45',
    'FM14': 'J-15',
    'FM15': 'J-52',
}


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
        flow_rate_threshold=2.0,
        pressure_threshold=2.0,
        )

    # Create a dictionary of dataframes
    dfs_dict = {
        'no_leak': dfs[0],
        'leak_1': dfs[1],
        'leak_2': dfs[2],
        'leak_3': dfs[3],
    }


    columns = dfs_dict['no_leak'].columns
    sensor_labels = [
        sensor_to_label[sensor[0:4]] for sensor in columns[0:(len(columns)//2)]
        ]
    num_sensors = len(sensor_labels)
    print(num_sensors)

    '''
    preprocessor = MinMaxScaler()
    preprocessor.fit(dfs_dict['no_leak'])

    for key, df in dfs_dict.items():
        dfs_dict[key] = preprocessor.transform(df)
    '''
        
    # Perform PCA on the data
    #pca = PCA(n_components=8)
    pca = KernelPCA(
        n_components=7, kernel="rbf", gamma=1, fit_inverse_transform=True, alpha=0.1
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

    
    wn = wntr.network.WaterNetworkModel('data/IISc_epanet.inp')

    # Get networkx graph
    length = wn.query_link_attribute('length')
    G = wn.to_graph(link_weight=length)
    G = G.to_undirected()

    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    id_to_label = nx.get_node_attributes(G, 'old_label')
    label_to_id = {v: k for k, v in id_to_label.items()}

    sensor_ids = [label_to_id[label] for label in sensor_labels]
    
    laplacian = sparse.csr_matrix(nx.laplacian_matrix(G, weight='weight'), dtype=np.float64)
    eigenvalues, eigenvectors = tf.linalg.eigh(laplacian.toarray())
    eigenvectors, eigenvalues = eigenvectors[:, :num_eigenpairs], eigenvalues[:num_eigenpairs]
    eigenvalues, eigenvectors = tf.convert_to_tensor(eigenvalues, dtype=dtype), tf.convert_to_tensor(eigenvectors, dtype)
    
    N = len(G)
    vertex_dim = 2
    point_kernel = gpflow.kernels.Matern32()
    kernel = GraphMaternKernel(
        (eigenvectors, eigenvalues), 
        nu=3/2, 
        kappa=5, 
        sigma_f=1, 
        vertex_dim=vertex_dim, 
        point_kernel=point_kernel, 
        dtype=dtype
    )
    '''
    kernel = GraphDiffusionKernel(
        (eigenvectors, eigenvalues), 
        kappa=5, 
        sigma_f=1, 
        vertex_dim=vertex_dim, 
        point_kernel=point_kernel, 
        dtype=dtype
    )

    '''
    for case in ['leak_1', 'leak_2', 'leak_3']:
        x_train = np.array(sensor_ids).reshape(-1, 1)
        y_train_1 = recon_error[case].values[0:100, num_sensors:].mean(axis=0).reshape(-1, 1)#dfs_dict['no_leak'].values[0, 9:].reshape(-1, 1)
        y_train_2 = recon_error[case].values[0:100, 0:num_sensors].mean(axis=0).reshape(-1, 1)#dfs_dict['no_leak'].values[0, 9:].reshape(-1, 1)
        y_train = np.concatenate((y_train_1, y_train_2), axis=1)

        x_train = tf.convert_to_tensor(x_train, dtype=dtype)
        y_train = tf.convert_to_tensor(y_train, dtype=dtype)

        model = gpflow.models.GPR(data=(x_train, y_train), kernel=kernel, noise_variance=0.01)

        optimize_GPR(model, 2500)
        gpflow.utilities.print_summary(model)

        x_test = np.arange(0, N).reshape(-1, 1)
        x_test = tf.convert_to_tensor(x_test, dtype=dtype)

        mean, var = model.predict_f(x_test)

        mean = mean.numpy()
        var = var.numpy()

        mean[:, 0] = mean[:, 0]/np.max(mean[:, 0])
        mean[:, 1] = mean[:, 1]/np.max(mean[:, 1])

        var[:, 0] = var[:, 0]/np.max(var[:, 0])
        var[:, 1] = var[:, 1]/np.max(var[:, 1])

        mean = mean.mean(axis=1)
        var = var.mean(axis=1)
        
        # Assign mean and variance to nodes
        for i, node in enumerate(G.nodes):
            G.nodes[node]['mean'] = mean[i]
            G.nodes[node]['var'] = var[i]

        sensor_nodes_pos = {}
        for node, _pos in pos.items():
            if node in sensor_ids:
                sensor_nodes_pos[node] = _pos

        sensor_X = np.zeros((num_sensors, 2))
        for i in range(0, num_sensors):
            sensor_X[i, 0] = sensor_nodes_pos[sensor_ids[i]][0]
            sensor_X[i, 1] = sensor_nodes_pos[sensor_ids[i]][1]

        leak_pipe_label = {
            'leak_1': ('J-32', 'J-86'),#, 'P-49'),#'P-2',
            'leak_2': ('J-44', 'J-35'),#), 'P-2'),#'P-49',
            'leak_3': ('J-15', 'J-72'),#, 'P-26'),#'P-26',
        }

        leak_pipe_id = {}
        for key, value in leak_pipe_label.items():
            leak_pipe_id[key] = (label_to_id[value[0]], label_to_id[value[1]])

        # Plot the graph with mean as the color
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.scatter(
            sensor_X[:, 0], 
            sensor_X[:, 1], 
            c='blue',
            s=500, 
            alpha=0.25,
        )
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color=mean, cmap='Reds', ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax, width=3)

        nx.draw_networkx_edge_labels(
            G,
            pos=pos,
            edge_labels={leak_pipe_id[case]: 'X'},
            font_size=16,
            font_color='blue',
        )
        
        plt.colorbar()

        plt.savefig(f'{case}.png', dpi=300)

        plt.show()
    

    


if __name__ == "__main__":
    main()