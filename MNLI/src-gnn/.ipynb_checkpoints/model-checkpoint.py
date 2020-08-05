# Dependencies
import torch
import torch.nn as nn
import math
#from transformers import BertModel

import config
from config import nli_config
import utils

"""
implement baseline first
glove
GAT
cross att
local comparison(F(h;p;h-p;h*p))
agrregate by mean and max
prediction
"""


# work
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree


# conv, GAT
"""
GATConv(in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, bias: bool = True, **kwargs)
"""

# conv, do later
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

# apply l layers of GraphConv
class GraphEncoder(nn.Module):
    def __init__(self, input_d=config.EMBEDDING_D, output_d = config.EMBEDDING_D, conv="gat", num_layers=2):
        super().__init__()
        self.dropout = nn.Dropout(p=config.DROUP_OUT_PROB)
        self.activation = nn.ReLU(inplace=True)
        self.num_layers = num_layers
        if conv == "hggcn":
            self.conv = None
        elif conv == "gat":
            # negative_slope is slope of LeakyRelu
            self.conv = GATConv(in_channels = input_d,
                                out_channels = input_d,
                                heads = 2, concat = True,
                                negative_slope = 0.2,
                                dropout = 0.0,
                                add_self_loops= True,
                                bias = True)
    def forward(batch):
        for l in range(self.num_layers):
            batch, att = self.conv(batch, return_attention_weights=None)
        return batch
        
        
class CrossAttentionLayer(nn.Module):
    """
    cross attention, similar to Decomp-Att
    but no fowrad nn, use Wk Wq Wv
    input: query vector(b*n*d), content vector(b*m*d)
    ouput: sof aligned content vector to query vector(b*n*d)
    """
    def __init__(self, input_d, output_d, hidden_d, number_of_head=1):
        super().__init__()
        self.dropout = nn.Dropout(p=config.DROUP_OUT_PROB)
        self.activation = nn.ReLU(inplace=True)
        # params
        self.hidden_size = hidden_size
        self.Wq = nn.Parameter(torch.Tensor(input_d, hidden_d))
        self.Wk = nn.Parameter(torch.Tensor(input_d, hidden_d))
        self.Wv = nn.Parameter(torch.Tensor(input_d, output_d))
        self.Wo = nn.Parameter(torch.Tensor(input_d, output_d))
        nn.init.xavier_uniform_(self.Wk, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.Wq, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.Wv, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.Wo, gain=nn.init.calculate_gain('relu'))
        
    def forward(self, h1, h2, mask=None):
        Q = torch.matmul(h1, self.Wq)
        #K = torch.matmul(h2, self.Wk)
        #Kt = torch.matmul(h2, self.Wk).permute(0,2,1)
        #E = torch.matmul(Q, Kt)
        K = torch.einsum("bnx,xy->bny", [h2, self.Wk])
        V = torch.matmul(h2, self.Wv)
        E = torch.einsum("bnd,bmd->bnm", [Q, K]) # batch, n/m, dimension
        if mask is not None:
            E = E.masked_fill(mask==0, float(-1e7))
        A = torch.softmax(E / (math.sqrt(self.cross_attention_hidden)), dim=2) #soft max dim = 2
        # attention shape: (N, heads, query_len, key_len)
        aligned_2_for_1 = torch.einsum("bnm,bmd->bnd", [A, V])
            
        return aligned_2_for_1
        
class SynNLI_Model(nn.Module):
    """
    word embedding (glove/bilstm/elmo/bert) + SRL embedding
    graph encoder (GAT/HGAT/HetGT...)
    cross attention allignment (CrossAtt)
    local comparison(F(h;p;h-p;h*p))
    aggregation ((tree-)LSTM?)
    prediction (FeedForward)
    """
    def __init__(self, nli_config=config.nli_config, pretrained_embedding_tensor=None):
        super().__init__()
        self.settings = nli_config
        d = self.settings.hidden_size
        # dropouts
        self.dropout = nn.Dropout(p=config.DROUP_OUT_PROB)
        self.activation = nn.ReLU(inplace=True)
        # embedding
        if(self.settings.embedding == "glove300d"):
            #pretrained_embedding_tensor = utils.load_glove_vector() should not be here
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding_tensor)
        else:
            self.embedding = nn.Embedding(config.GLOVE_VOCAB_SIZE, config.GLOVE_DIMENSION)
        # encoder
        if self.settings.encoder == None:
            self.encoder = None
        else:
            self.encoder = GraphEncoder(conv=self.settings.encoder)
        # cross_att
        if(self.settings.cross_att == "scaled_dot"):
            self.cross_att = CrossAttentionLayer()
        # local comp fnn h, p^, h-p^, h*p^
        self.local_comp = nn.Sequential(nn.Linear(4*d, d), nn.ReLU(), nn.Linear(d,d))
        # aggregation is max
        # self.aggr = (partial)torch.max(dim=1, keep_dim=False)
        # cls
        self.classifier = nn.Linear(d, config.NUM_CLASSES)
        self.criterion = nn.BCEWithLogitsLoss()
    
    
    def forward_nn(self, batch):
        """
        'sentence1' : {'input_ids', 'token_type_ids', 'attention_mask'} batch*len*d
        'sentence2' :  {'input_ids', 'token_type_ids', 'attention_mask'}
        'gold_label' : batch*1
        """
        # alias
        lf = config.lf
        hf = config.hf
        pf = config.pf
        # get embedding
        batch[lf] = 
        # get graph contextualized embedding
        hh, poolh = self.encoder()
        hp, poolp = self.encoder()
        # soft alignment, not considering mask...
        mh = attention_mask=batch[config.h_field]['attention_mask']
        mp = attention_mask=batch[config.p_field]['attention_mask']
        maskph = torch.einsum("bn,bm->bnm", [mh, mp])
        maskhp = torch.einsum("bn,bm->bnm", [mp, mh])
        # (b, l_h, d)
        p_hat = self.cross_attention(hh, hp, maskph) # b * l_h * d
        # aligned_h_for_p = self.cross_attention(hp, hh, maskhp) # b * l_p *d
        
        # comparison stage
        # (b, l_h, d)
        cmp_hp = self.fnn(torch.cat((p_hat, hh, p_hat-hh, p_hat*hh), dim=2))
        #cmp_ph = self.fnn(torch.cat((aligned_h_for_p, hp), dim=2))
        
        # aggregatoin stage (mean + max for h part IMO)
        # (b, d)
        sent_hp_max = torch.max(cmp_hp, dim=1, keepdim=False)[0] # maxpool
        sent_hp_mean = torch.mean(cmp_hp, dim=1, keepdim=False) # meanpool
        
        #sent_ph = torch.sum(cmp_ph, dim=1, keepdim=False)
        # prediction get
        #logits = self.classifier(torch.cat((sent_hp, sent_ph), dim=1))
        logits = self.classifier(torch.cat((sent_hp_max, sent_hp_mean), dim=1))
        logits = logits.squeeze(-1)
        return logits
    
    # the nn.Module method
    def forward(self, batch):
        logits = self.forward_nn(batch)
        batch[config.label_field] = batch[config.label_field].to(dtype=torch.float)
        loss = self.criterion(logits, batch[config.label_field])
        return loss, logits
    
    # return sigmoded score
    def _predict_score(self, batch):
        logits = self.forward_nn(batch)
        scores = torch.sigmoid(logits)
        scores = scores.detach().cpu()
        return scores
    
    # return True False based on score + threshold
    def _predict(self, batch, threshold=0.5):
        scores = self._predict_score(batch)
        return torch.argmax(scores, dim=1) # return highest