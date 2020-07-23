

"""
old collate

for f in [config.h_field, config.p_field]:
    tokens_tensors = [s[f] for s in samples]
    segments_tensors = [torch.tensor([0] * len(s[f]), dtype=torch.long) for s in samples]
    # zero pad to same length
    tokens_tensors = pad_sequence(tokens_tensors,  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,  batch_first=True)
    # attention masks, set none-padding part to 1 for LM to attend
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill( tokens_tensors != 0, 1)
    batch[f] = {
        "tokens_tensors" : tokens_tensors,
        "segments_tensors" :  segments_tensors,
        "masks_tensors" : masks_tensors
    }
"""

##############################################################
# for bert embedding, self tokenized                                         #
##############################################################

# note, this should be called after tokenizer is assigned
def sent_to_tensor(s, tokenizer):
    assert(tokenizer is not None)
    
    tokens = tokenizer.tokenize(s)
    if(len(tokens) > config.BERT_MAX_INPUT_LEN-2):
        if config.DEBUG:
            print("a sentence: \n" + s + "\n is truncated to fit max bert len input size.")
        tokens = tokens[:BERT_MAX_INPUT_LEN-2]
    tokens.insert(0, "[CLS]")
    tokens.append("SEP")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    tensor = torch.tensor(ids)
    return tensor

# for visualization 
def tensor_to_sent(t, tokenizer):
    assert(tokenizer is not None)
    tokens = tokenizer.convert_ids_to_tokens(t)
    sent = " ".join(tokens)
    return sent

# for CrossBERT dataset
def CrossBERT_preprocess(raw_data, tokenizer=None):
    #tokenizer
    if tokenizer == None:
        tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)
    else:
        tokenizer = tokenizer 
    
    # filed alias
    pf = config.p_field
    hf = config.h_field
    lf = config.label_field
    
    processed_data = list()
    maxlen = {pf: 0, hf : 0}
    
    ## to tensor and get maxlen
    for instance in raw_data:
        # label
        l = instance[lf]
        if(l not in config.label_to_id.keys()):
            continue
        # storage
        processed_instance = {
            pf: sent_to_tensor(instance[pf], tokenizer),
            hf: sent_to_tensor(instance[hf], tokenizer),
            lf: torch.tensor(config.label_to_id[l], dtype=torch.long)
        }
        processed_data.append(processed_instance)
    
    # padding no here
    return processed_data


class MNLI_CrossBERT_Dataset(Dataset):
    """
    MNLI set for CrossBERT baseline
    source: 
    wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip @ 2020/7/21 17:09
    self.j_data is list of jsons
    self.raw_data is list of (hyposesis, premise, gold label)
    # self.tensor_data is list of tensored data (generate by sent_to_tensor for bert)
    """
    def __init__(self,
                 file_path=config.DEV_MA_FILE,
                 mode="develop",
                 process_fn=CrossBERT_preprocess,
                 tokenizer=None,
                 data_config=config.data_config,
                 save=False):
        # super(MNLI_CrossBERT_Dataset, self).__init__()
        # decide config
        self.mode = mode
        
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)
        else:
            self.tokenizer = tokenizer 
        
        # load raw data
        self.file_path = file_path
        with open(self.file_path) as fo:
            self.raw_lines = fo.readlines()
        # to json
        self.j_data = [json.loads(line) for line in self.raw_lines]
        self.tensor_data = process_fn(self.j_data, self.tokenizer)
        return None
        
        
    def __getitem__(self, index):
        return self.tensor_data[index]
        
    def __len__(self):
        return len(self.tensor_data)
    
##############################################################
