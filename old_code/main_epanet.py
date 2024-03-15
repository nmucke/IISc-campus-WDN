import pdb
import wntr
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def main():

    # Load epanet model
    wn = wntr.network.WaterNetworkModel('epanet_input_files/IISc_Epanet_Revised281123.inp')

    # Solve epanet model
    sim = wntr.sim.EpanetSimulator(wn)

    results = sim.run_sim()

    # Get networkx graph
    G = wn.to_graph()


    pos = nx.get_node_attributes(G, 'pos')

    # Get edge weights
    length = wn.query_link_attribute('length')

    plt.figure()
    for i in range(results.node['pressure'].values.shape[1]):
        plt.plot(results.node['pressure'].values[:, i])
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (m)')
    plt.show()
    

    plt.figure(figsize=(20, 20))
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=100,
        node_color=results.node['pressure'].values[30],
        edge_color=results.link['flowrate'].values[30],
        width=4.,
        with_labels=False,
    )
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap='viridis', 
        norm=plt.Normalize(vmin=results.node['pressure'].values[30].min(), vmax=results.node['pressure'].values[30].max()),
        )
    plt.colorbar(sm, label='Pressure (m)')

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap='viridis', 
        norm=plt.Normalize(vmin=results.link['flowrate'].values[30].min(), vmax=results.link['flowrate'].values[30].max()),
        )
    plt.colorbar(sm, label='Flow rate (m3/s)')
    
    plt.title('IISc Water Network')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()




if __name__ == '__main__':
    main()