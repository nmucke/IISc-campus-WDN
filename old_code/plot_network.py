


import networkx as nx
import numpy as np
import wntr
import pdb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar


# Set font size in matplotlib
plt.rcParams.update({'font.size': 30})

def main():

    leak_pipe = {
        'Leak 1': ('J-32', 'J-86'),#, 'P-49'),#'P-2',
        'Leak 2': ('J-44', 'J-35'),#), 'P-2'),#'P-49',
        'Leak 3': ('J-15', 'J-72'),#, 'P-26'),#'P-26',
    }
    # Load epanet model
    wn = wntr.network.WaterNetworkModel('epanet_input_files/IISc_Epanet_Revised281123.inp')

    # Get networkx graph
    G = wn.to_graph()
    G = G.to_undirected()

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

     # Plot the sensor nodes
    plt.figure(figsize=(15, 20))
    
    plt.grid()
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=200,
        node_color='k',
        edge_color='k',
        width=4.,
        with_labels=False,
    )

    reservoir_coords = {
        'STP 1': (401.47, 1263.50),
        'STP 2': (1399.24, 989.76),
    }
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
             
    plt.plot(
        sensor_X[:, 0], 
        sensor_X[:, 1], 
        '.',
        markersize=50,
        color='tab:blue',
        label='Sensor nodes',
    )
    for (case, color) in zip(
        ['Leak 1', 'Leak 2', 'Leak 3'],
        ['tab:red', 'tab:orange', 'tab:green'],
    ):
        leak_node_1 = leak_pipe[case][0]
        leak_node_2 = leak_pipe[case][1]

        leak_node_1_pos = pos[leak_node_1]
        leak_node_2_pos = pos[leak_node_2]

        # Get coordinates of midpoint between the two nodes
        midpoint = np.zeros((2,))
        midpoint[0] = (leak_node_1_pos[0] + leak_node_2_pos[0])/2.
        midpoint[1] = (leak_node_1_pos[1] + leak_node_2_pos[1])/2.
        

        plt.plot(
            midpoint[0], 
            midpoint[1], 
            'X',
            markersize=40,
            color=color,
            label=case,
        )
        #if case == 'Leak 1':
        #    plt.text(midpoint[0]-120, midpoint[1]-10, case, fontsize=30)
        #elif case == 'Leak 2':
        #    plt.text(midpoint[0]-120, midpoint[1]-15, case, fontsize=30)
        #else:
        #    plt.text(midpoint[0]-40, midpoint[1]+30, case, fontsize=30)

    
    for (reservoir, color) in zip(
        ['STP 1', 'STP 2'],
        ['tab:brown', 'tab:purple'],
    ):
        reservoir_pos = reservoir_coords[reservoir]

        plt.plot(
            reservoir_coords[reservoir][0], 
            reservoir_coords[reservoir][1], 
            's',
            markersize=40,
            color=color,
            label=reservoir,
        )
        #if reservoir == 'STP 1':
        #    plt.text(reservoir_pos[0]-100, reservoir_pos[1]-10, reservoir, fontsize=30)
        #else:
        #    plt.text(reservoir_pos[0]+30, reservoir_pos[1], reservoir, fontsize=30)

    scalebar = ScaleBar(1., location=3)
    plt.gca().add_artist(scalebar)

    scalebar = ScaleBar(1., location=3, rotation='vertical')
    plt.gca().add_artist(scalebar)

    
    
    plt.title('IISc Water Network')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.savefig(f'network.eps')
    plt.show()


    return 0



if __name__ == '__main__':
    main()