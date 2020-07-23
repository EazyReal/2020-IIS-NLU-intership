import os
from pathlib import Path

# whether log when executing
DEBUG = True
LOG = True

# Paths 
SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PROJ_ROOT = SRC_ROOT.parent

DATA_ROOT = PROJ_ROOT / "data" / "multinli_1.0"
PARAM_PATH = PROJ_ROOT / "param"

PDATA_ROOT = PROJ_ROOT / "data" / "preprocessed_MNLI"

DEV_MMA_FILE = DATA_ROOT / "multinli_1.0_dev_mismatched.jsonl"
DEV_MA_FILE = DATA_ROOT / "multinli_1.0_dev_matched.jsonl"
TRAIN_FILE = DATA_ROOT / "multinli_1.0_train.jsonl"

PDEV_MMA_FILE = DATA_ROOT / "pre_multinli_1.0_dev_mismatched.jsonl"
PDEV_MA_FILE = DATA_ROOT / "pre_multinli_1.0_dev_matched.jsonl"
PTRAIN_FILE = DATA_ROOT / "pre_multinli_1.0_train.jsonl"

# Preprocssing / Data Config
data_config = {
    "file_path" : DEV_MA_FILE # this should be a param
}
label_to_id = {
    "contradiction" : 0,
    "neutral" : 1,
    "entailment" : 2,
}
h_field = "sentence2"
p_field = "sentence1"
label_field = "gold_label"

# MODEL
DEFAULT_USE_WEIGHTED_BCE = True


# Bert Enbedding
BERT_EMBEDDING = "bert-base-uncased" #cased?
BERT_MAX_INPUT_LEN = 512

tokenizer = BertTokenizer.from_pretrained(config.BERT_EMBEDDING)


# Trainning
BATCH_SIZE = 8
NUM_EPOCHS = 6
LR = 0.00001 # 1e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NUM_WARMUP = 100