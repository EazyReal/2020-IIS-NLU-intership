from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os
from collections import defaultdict
from sklearn import metrics

from preprocess import * 
from model import * 
import config as config

# flag for printing debug message or general message
TEST = True
LOG = True

# training params  in config

# Load Data using dataset.py

class CrossBERT_Trainer:
    # init trainer, if not specify model, will create one with weighted BCE with dataset when call train
    def __init__(self, model=None):
        #self.train_set = train_set
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def eval_model_when_training(self, model, dev_set):
        """
        when entering this function 
        model should be in the same device as trainer do
        """
        #model.to(self.device)
        model.eval()
        
        dev_loader = DataLoader(dev_set, batch_size=config.BATCH_SIZE, collate_fn=create_mini_batch)
        pred = []
        label = []
        
        for batch_i, batch in enumerate(tqdm(dev_loader)):
            label.extend(torch.argmax(batch[config.label_field].cpu(), dim = 1).numpy().tolist()) # label, one hot to tensor, shape = (batch) now
            #print(label.size())
            for k in batch:
                batch[k] = batch[k].to(self.device)
            pred_batch = model._predict(batch).numpy().tolist()
            pred.extend(pred_batch)
            
        report = {
            'report' : metrics.classification_report(pred, label, output_dict=True),
            'confusion_matrix' : metrics.confusion_matrix(pred, label)
        }
            
        return report
    
    def train(self, train_set=None, dev_set=None):
        
        # train set loading
        if train_set == None:
            if(config.LOG):
                print('loading trainset from {}'.format(config.TRAIN_FILE))
            train_set = MNLI_Raw_Dataset(config.TRAIN_FILE, mode="train")
        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, collate_fn=create_mini_batch)
        
        # dev set loading, id2qid
        if dev_set == None:
            if(config.LOG):
                print('loading devset from {}'.format(config.DEV_MA_FILE))
            dev_set = MNLI_Raw_Dataset(config.DEV_MA_FILE, mode="dev")
        dev_loader = DataLoader(dev_set, batch_size=config.BATCH_SIZE, collate_fn=create_mini_batch)
        
        bert_encoder = BertModel.from_pretrained(config.BERT_EMBEDDING)
        
        # model initialization if = None, will aplly pos_weight to BCE
        if self.model == None:
            if(config.LOG):
                print('initializing new model')
            self.model = CrossBERTModel(bert_encoder=bert_encoder)
        self.model.to(self.device)
        
        if TEST or LOG:
            print("model in cuda?", next(self.model.parameters()).is_cuda)
            
        # set saving directory
        model_file_path = "CrossBERT_baseline"
        save_model_path = config.PARAM_PATH / model_file_path

        if not os.path.exists(save_model_path):
            if config.LOG:
                print("directory {} doesn't exist, creating...".format(save_model_path))
            os.mkdir(save_model_path)
        if config.LOG:
            print("trained_model will be in {}".format(save_model_path))
            
        # optimizer
        if config.DEBUG:
            print(self.model.named_parameters())
        # weight decay scheduler setting
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.WEIGHT_DECAY},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = len(train_loader) * config.NUM_EPOCHS
        
        # AdamW
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.LR)
        # scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config.NUM_WARMUP,
                                                    num_training_steps=num_train_optimization_steps)
            
        # alias when migrating code
        model = self.model
        device = self.device
        
        if config.LOG:
            print('start training ... ')

        stat = {
            'report' : [],
            'confusion_matrix' : []
        }

        for epoch_i in range(config.NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for batch_i, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                
                for k in batch:
                    batch[k] = batch[k].to(device)
                #batch = [data.to(device) for key in batch] # batch[0] = ids, batch[1] = ...
                loss, logits = model(batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM) 
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                
            learning_rate_scalar = scheduler.get_lr()[0]
            print('lr = %f' % learning_rate_scalar)
            print('epoch %d train_loss: %.3f' % (epoch_i, running_loss / len(train_loader)))

            eval_epoch_frequency = 1
            if epoch_i % eval_epoch_frequency == 0:
                epoch_stat = self.eval_model_when_training(model, dev_set)
                for k in stat:
                    stat[k].append(epoch_stat[k])
                print('epoch %d eval_precision: %.3f eval_recall: %.3f eval_f1: %.3f' % 
                      (epoch_i, epoch_stat['report']['macro avg']['precision'], epoch_stat['report']['macro avg']['recall'], epoch_stat['report']['macro avg']['f1-score']))
                model_to_save = model
                torch.save(model_to_save.state_dict(),
                            str(save_model_path / "model_epoch{0}_precision:{1:.3f}_recall:{2:.3f}_f1:{3:.3f}_acc:{4:.3f}.m".
                                   format(epoch_i, epoch_stat['report']['macro avg']['precision'], epoch_stat['report']['macro avg']['recall'], epoch_stat['report']['macro avg']['f1-score'], epoch_stat['report']['accuracy'])))
        model.eval()
        return model, stat