import torch
import os
import logging
import config
from argparse import ArgumentParser
import stanza

###########################################################################################



###########################################################################################


# alias
p = config.pf
h = config.hf
l = config.lf

def print_compact_dep(doc):
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    return

def get_sent_repr(sent, parser, emb):
    

def preprocess_data(data_file=config.DEV_MA_FILE, emb_file=config.GLOVE, target=config.PDEV_MA_FILE):
    
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