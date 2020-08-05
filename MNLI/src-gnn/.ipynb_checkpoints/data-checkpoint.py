# internal
import config
import utils

# external
import json
import torch
from torch_geometric.data.data import Data
from stanza.models.common.doc import Document

# load json data from preprocessed file
def load_jdata(data_file, function_test=False):
    # data_file_loading
    with open(data_file) as fo:
        raw_lines = fo.readlines()
        jdata = [json.loads(line) for line in raw_lines]
    return jdata

# bulid GraohData
class GraphData(Data):
    """
    item is a raw json of parsed result
    """
    def __init__(self, data, word2idx, tolower=True):
        super(GraphData, self).__init__()
        g_p = utils.doc2graph(Document(data[config.pf]))
        g_h = utils.doc2graph(Document(data[config.hf]))
        self.edge_index_p = g_p.edge_index
        #print(g_p.node_attr)
        self.x_p = torch.tensor([ word2idx[w.lower()] for w in g_p.node_attr], dtype=torch.long)
        self.edge_index_h = g_h.edge_index
        self.x_h = torch.tensor([ word2idx[w.lower()] for w in g_h.node_attr], dtype=torch.long)
        self.label = data[config.lf]
        self.pid = data[config.idf]
    def __inc__(self, key, value):
        if key == 'edge_index_p':
            return self.x_p.size(0)
        if key == 'edge_index_h':
            return self.x_h.size(0)
        else:
            return super(GraphData, self).__inc__(key, value)
"""
# collate_fn is implemented by pytorch geo
# we need to add follow_batch=[] to handle batch
# usage  = Loader
loader = DataLoader(dev_data_set, batch_size=3, follow_batch=['x_p', 'x_h'])
batch = next(iter(loader))
"""