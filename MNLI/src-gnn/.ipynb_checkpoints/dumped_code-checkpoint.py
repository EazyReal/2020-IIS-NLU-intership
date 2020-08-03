"""
import networkx as nx
from torch_geometric.utils.convert import to_networkx
def draw(data):
    G = to_networkx(data)
    nx.draw(G)
    plt.savefig("path.png")
    plt.show()
"""