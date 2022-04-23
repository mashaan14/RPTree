import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plotGraph(adj_mat_plot):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    G_1 = nx.from_numpy_matrix(adj_mat_plot)
    pos = np.genfromtxt('data/ring238_Instances.csv', delimiter=",")
    nx.draw(G_1, pos, node_size=20)
    nx.draw(G_1, pos, node_size=20, alpha=0.75);
    plt.savefig('plot.png', dpi=600, bbox_inches='tight')