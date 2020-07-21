################################################################################
# Desciption

# author = ytlin
# decription = main program for preprocess + load model + train + eval + use
################################################################################

################################################################################
# Usage
"""
pip install -r requirements.txt
python -m main.py \
  --data_dir=... \
  --output_dir=... \
  --model_name=... \
  --model_file=... \
  --model_check_point=...\
  --model_config_file=... \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length=128 \
  --optimizer=adamw \
  --task_name=MNLI \
  --warmup_step=1000 \
  --learning_rate=3e-5 \
  --train_step=10000 \
  --save_checkpoints_steps=100 \
  --train_batch_size=128
"""
################################################################################


################################################################################
# Dependencies
################################################################################

# project
import config
import preprocessing
import model
import train
import evaluation
# external
import argparse


################################################################################
# ARGUMENTS
################################################################################
def init_args(arg_string=None):
    parser = argparse.ArgumentParser()

    # FILE PATHS
    parser.add_argument('--model_check_point', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='roberta_model')
    
    # ACTION
    parser.add_argument('--do_train', type=lambda x: (x.lower() == 'true'),
                        default=False)
    parser.add_argument('--train_file', nargs='*', type=str)
    parser.add_argument('--do_eval', type=lambda x: (x.lower() == 'true'),
                        default=False)
    parser.add_argument('--eval_file', nargs='*', type=str)
    parser.add_argument('--do_predict', type=lambda x: (x.lower() == 'true'),
                        default=False)
    parser.add_argument('--predict_file', nargs='*', type=str)
    # MODEL PARAMETERS
    parser.add_argument('--max_seq_length', type=int, default=512)
    # DATA PREPROCESSING
    parser.add_argument('--sup_evidence_as_passage', action='store_true')
    parser.add_argument('--max_window_slide_dist', type=int, default=128)
    # PARAMETERS FOR TRAINING
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--train_batch_size', type=int, default=6)
    parser.add_argument('--train_epochs', type=int, default=4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--save_epochs', type=int, default=1)
    # PARAMETERS FOR PREDICTING
    parser.add_argument('--predict_batch_size', type=int, default=6)
    # OTHERS
    parser.add_argument('--err_to_dev_null', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)

    args = parser.parse_args(arg_string)

    # DEFAULT TO THE MULT-LABEL MODE
    if args.do_train and len(args.train_file) == 0:
        raise ValueError('"do_train" is set but no "train_file" is given.')
    if args.do_eval and len(args.eval_file) == 0:
        raise ValueError('"do_eval" is set but no "eval_file" is given.')
    if args.do_predict and len(args.predict_file) == 0:
        raise ValueError('"do_predict" is set but no "predict_file" is given.')

    model_config_path = os.path.join(
        args.model_name_or_path, ARGS_FILE_NAME)
    if os.path.exists(model_config_path):
        with open(model_config_path) as f:
            model_config = json.load(f)
        for key, val in model_config.items():
            setattr(args, key, val)

    # DEVICE SETTING
    if torch.cuda.is_available() and not args.force_cpu:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # ERROR STREAM
    if args.err_to_dev_null:
        args.err_stream = open(os.path.devnull, mode='w')
    else:
        args.err_stream = sys.stderr

    return args

################################################################################
# THE MAIN FUNCTION
################################################################################
def main():
    args = init_args()
    print('CREATING TOKENIZER...', file=args.err_stream)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    print('CREATING MODEL...', file=args.err_stream)
    model = YesNoModel(args)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device=args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        train_data = pd.DataFrame()

        if args.train_file[0].endswith(".tsv"):
            train_data = pd.read_csv(args.train_file[0], sep='\t', header=0, index_col='id')
        elif args.train_file[0].endswith(".csv"):
            train_data = pd.read_csv(args.train_file[0], sep=',', header=0, index_col='id')
        
        print('TRAINING...', file=args.err_stream)
        train(args, model, tokenizer, train_data)

    if args.do_eval:
        eval_data = pd.DataFrame()

        if args.eval_file[0].endswith(".tsv"):
            eval_data = pd.read_csv(args.eval_file[0], sep='\t', header=0, index_col='id')
        elif args.eval_file[0].endswith(".csv"):
            eval_data = pd.read_csv(args.eval_file[0], sep=',', header=0, index_col='id')
        
        print('EVALUATING...', file=args.err_stream)
        #eval_result, correct_qid, error_qid= eval(args, model, tokenizer, eval_data)
        eval_result, error_qid= eval(args, model, tokenizer, eval_data)
        #eval_result = eval(args, model, tokenizer, eval_data)
        
        print(eval_result)
        #print("correct: ", correct_qid)
        print("error: ", error_qid)


    if args.do_predict:
        predict_data = pd.DataFrame()

        if args.predict_file[0].endswith(".tsv"):
            predict_data = pd.read_csv(args.predict_file[0], sep='\t', header=0, index_col='id')
        elif args.predict_file[0].endswith(".csv"):
            predict_data = pd.read_csv(args.predict_file[0], sep=',', header=0, index_col='id')
        
        print('PREDICTING...', file=args.err_stream)
        final_predictions = predict(args, model, tokenizer, predict_data)

        print('WRITING PREDICTIONS...', file=args.err_stream)
        prediction_file_path = os.path.join(
            args.output_dir, 'predictions.json')
        with open(prediction_file_path, mode='w') as f: #modified by ytlin 
            json.dump(final_predictions, f, ensure_ascii=False, indent=4)

    


if __name__ == "main":
    main()