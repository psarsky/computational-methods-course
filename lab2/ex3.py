import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def load_graph(file):
    G = nx.Graph()
    with open(file, 'r') as f:
        for line in f:
            u, v, r = map(float, line.split())
            G.add_edge(int(u), int(v), weight=r)
    return G

def kirchhoff_analysis(G, source, target, E):
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for u, v, data in G.edges(data=True):
        r = data['weight']
        i_u, i_v = node_index[u], node_index[v]
        
        A[i_u, i_u] += 1 / r
        A[i_v, i_v] += 1 / r
        A[i_u, i_v] -= 1 / r
        A[i_v, i_u] -= 1 / r
    
    b[node_index[source]] = E
    b[node_index[target]] = -E
    
    potentials = lstsq(A, b)[0]
    
    currents = {}
    for u, v, data in G.edges(data=True):
        r = data['weight']
        i_u, i_v = node_index[u], node_index[v]
        I = (potentials[i_u] - potentials[i_v]) / r
        currents[(u, v)] = I
    
    return currents

def visualize_graph(G, currents):
    pos = nx.spring_layout(G)
    edges, weights = zip(*currents.items())
    nx.draw(G, pos, with_labels=True, edge_color=list(weights), edge_cmap=plt.cm.coolwarm)
    plt.show()

def main():
    G = load_graph("grid_graph.txt")
    source, target, E = 0, 5, 10
    currents = kirchhoff_analysis(G, source, target, E)
    visualize_graph(G, currents)

if __name__ == "__main__":
    main()