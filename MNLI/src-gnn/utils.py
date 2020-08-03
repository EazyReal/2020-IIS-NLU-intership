import torch
import os
import logging
import config
from argparse import ArgumentParser
import stanza

import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import config
import utils
import stanza
import numpy as np
import pickle

###########################################################################################



###########################################################################################


# alias
p = config.pf
h = config.hf
l = config.lf


def draw(data, node_size=1000, font_size=12, save_img_file=None):
    """
    input: (torch_geometric.data.data.Data, path or string)
    effect: show and save data
    """
    G = to_networkx(data)
    pos = nx.nx_pydot.graphviz_layout(G)
    if(data.edge_attr != None):
        edge_labels = {(u,v):lab for u,v,lab in data.edge_attr}
    if(data.node_attr != None):
        node_labels = dict(zip(G.nodes, data.node_attr))
    nx.draw(G, pos=pos, nodecolor='r', edge_color='b', node_size=node_size, with_labels=False)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=font_size)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=font_size)
    print(G.nodes)
    print(G.edges)
    if save_img_file != None:
        plt.savefig(save_img_file)
    plt.show()

    
def text2dep(s, nlp):
    """
    2020/8/3 18:30
    input (str:s, StanzaPipieline: nlp), s is of len l
    output (PytorchGeoData : G)
    G = {
     x: id tensor
     edge_idx : edges size = (2, l-1)
     edge_attr: (u, v, edge_type in str)
     node_attr: text
    }
    """
    doc = nlp(s)
    # add root token for each sentences
    x = torch.tensor(list(range(doc.num_tokens+len(doc.sentences))))
    #y = torch.tensor(list(range(doc.num_tokens+len(doc.sentences))))
    e = [[],[]]
    edge_info = []
    node_info = []
    prev_token_sum = 0
    prev_root_id = 0
    cur_root_id = 0
    # get original dependency
    for idx, sent in enumerate(doc.sentences):
        sent.print_dependencies
        # node info by index(add root at the beginning of every sentence)
        cur_root_id = len(node_info)
        node_info.append("<ROOT>")
        for token in sent.tokens:
            node_info.append(token.to_dict()[0]['text'])
        # edge info by index of u in edge (u,v)
        for dep in sent.dependencies:
            id1 = prev_token_sum + int(dep[0].to_dict()["id"])
            id2 = prev_token_sum + int(dep[2].to_dict()["id"])
            e[0].append(id1)
            e[1].append(id2)
            edge_info.append((id1, id2, dep[1]))
        prev_token_sum += len(sent.tokens)+1
        # add links between sentence roots
        if(cur_root_id != 0):
            id1 = prev_root_id
            id2 = cur_root_id
            e[0].append(id1)
            e[1].append(id2)
            edge_info.append((id1, id2, "bridge"))
        prev_root_id = cur_root_id
    # done building edges and nodes
    e = torch.tensor(e)
    G = Data(x=x, edge_index=e, edge_attr=edge_info, node_attr=node_info)
    return G

def print_compact_dep(doc):
    """
    input: stanza Doc
    effect: show dependency edges
    """
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    return

def get_sent_repr(sent, parser, emb):
    pass
    

def load_glove_vector(glove_embedding_file = config.GLOVE, dimension=config.GLOVE_DIMENSION, save_vocab = config.GLOVE_VOCAB, save_word2id = config.GLOVE_WORD2ID, save_dict=True):
    words = []
    idx = 0
    word2idx = {}
    glove = []
    #glove = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(glove_embedding_file, 'r') as fo:
        lines = fo.readlines()
        # add [UNK] handler
        words.append("[UNK]")
        word2idx["[UNK]"] = idx
        glove.append(np.zeros(300)) # 300 is vector dimension
        idx += 1
        # load vectors
        for line in tqdm(lines):
            line = line.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            vector = np.asarray(line[1:], "float32")
            glove.append(vector)
            idx += 1
    #glove = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    #glove.flush()
    if save_dict == True:
        pickle.dump(words, open(config.GLOVE_ROOT / config.GLOVE_VOCAB, 'wb'))
        pickle.dump(word2idx, open(config.GLOVE_ROOT / config.GLOVE_WORD2ID, 'wb'))
    return glove, words, word2idx, idx

def preprocess_data(data_file=config.DEV_MA_FILE, emb_file=config.GLOVE, target=config.PDEV_MA_FILE):
    """
    input (data = str, embedding = str, target file = str)
    effect preprocess and save data to target
    return preprocessed data
    """
    # alias
    p = config.pf
    h = config.hf
    l = config.lf
    
    # stanza dinit
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    
    # data_file_loading
    with open(data_file) as fo:
        raw_lines = fo.readlines()
        json_data = [json.loads(line) for line in raw_lines]
        
    # glove embedding loading
    glove = {}
    with open(emb_file, 'r', encoding="utf-8") as fo:
        lines = fo.readlines()
        for line in tqdm(lines):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove[word] = vector
            
    # preprocessing
    for data in json_data:
        # only add those who have 
        if(data[l] not in config.label_to_id.keys()):
            continue
        
    
    return