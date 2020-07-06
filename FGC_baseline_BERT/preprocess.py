# dependencies
import json
import torch
import pandas as pd

from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from pathlib import Path

# local settings etc
PRETRAINED_MODEL_NAME = "bert-base-chinese"
# data_dir = './FGC_release_1.7.13/'
# data_file = data_dir + 'FGC_release_all_dev.json'

# obtain pretrained bert tokenizer(init?)
# PRETRAINED_MODEL_NAME = "bert-base-chinese"
# tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# custom dataset for FGC data
class FGC_Dataset(Dataset):
    """
        FGC release all dev.json
        usage FGC_Dataset(file_path, mode, tokenizer)
        for tokenizer:
            PRETRAINED_MODEL_NAME = "bert-base-chinese"
            tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
        for file_path:
            something like ./FGC_release_1.7.13/FGC_release_all_dev.json
        for mode:
            ["train", "develop", "test"]
    """
    # read, preprocessing
    def __init__(self, data_file_ref, mode, tokenizer=BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)):
        # load raw json
        assert mode in ["train", "develop", "test"]
        self.mode = mode
        with open(data_file_ref) as fo:
            self.raw_data = json.load(fo)
        self.tokenizer = tokenizer 
        self.tokenlized_pair = None
        
        # generate raw pairs of q sent s
        self.raw_pair = list()
        for instance in self.raw_data:
            q = instance["QUESTIONS"][0]["QTEXT_CN"]
            sentences = instance["SENTS"]
            for idx, sent in enumerate(sentences):
                # check if is supporting evidence
                lab = idx in instance["QUESTIONS"][0]["SHINT_"]
                self.raw_pair.append((q, sent["text"], lab))
        
        # generate tensors 
        self.dat = list()
        for instance in self.raw_pair:
            q, sent, label = instance
            
            if mode is not "test":
                label_tensor = torch.tensor(label)
            else:
                label_tensor = None
            
            # first sentence, use bert tokenizer to cut subwords
            subwords = ["[CLS]"]
            q_tokens = self.tokenizer.tokenize(q)
            subwords.extend(q_tokens)
            subwords.append("[SEP]")
            len_q = len(subwords)
            
            # second sentence
            sent_tokens = self.tokenizer.tokenize(sent)
            subwords.extend(sent_tokens)
            subwords.append("[SEP]")
            len_sent = len(subwords)
            
            # subwords to ids, ids to torch tensor
            ids = self.tokenizer.convert_tokens_to_ids(subwords)
            tokens_tensor = torch.tensor(ids)
            
            # segments_tensor
            segments_tensor = torch.tensor([0] * len_q + [1] * len_sent, dtype=torch.long)
            self.dat.append((tokens_tensor, segments_tensor, label_tensor))
            
        return None
    
    # get one data of index idx
    def __getitem__(self, idx):
        return self.dat[idx]
    
    def __len__(self):
        return len(self.dat)