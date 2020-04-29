# data2text-three-dimensions

This repo contains code for [Table-to-Text Generation with Effective Hierarchical Encoder on Three Dimensions (Row, Column and Time)](https://www.aclweb.org/anthology/D19-1310.pdf) (Gong, H., Feng, X., Qin, B., & Liu, T.; EMNLP 2019); this code is based on [data2text-plan-py](https://github.com/ratishsp/data2text-plan-py).


## Requirement

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Note that requirements.txt contains necessary version requirements of certain dependencies and Python version is 3.6.

## data and model

Before executing commands in the following sections, the data and/or trained model need to be downloaded and extracted. They are available as a single tar.gz file at link https://www.dropbox.com/s/hnij20b0rm5dn9b/data_and_model.tar.gz?dl=0 (suited for global user) or https://pan.baidu.com/s/11mXCQnWFW9TLM2O3bfeHnQ (retrieval code: cqxx ). Please move the extracted folder rotowire and paper_model into this repo's folder.

## preprocessing

The following command will preprocess the data.

```
BASE=./rotowire

python preprocess.py -train_src $BASE/src_train.txt -train_src_hist $BASE/hist_full/src_train_hist_3.txt  -train_tgt $BASE/tgt_train.txt -valid_src $BASE/src_valid.txt -valid_src_hist $BASE/hist_full/src_valid_hist_3.txt -valid_tgt $BASE/tgt_valid.txt -save_data $BASE/preprocess/roto-encdec-history-window-3 -src_seq_length 10000 -tgt_seq_length 1000 -dynamic_dict -train_ptr $BASE/enc-dec-train-roto-ptrs.txt
```

## training

The following command will train the model.

```
BASE=./rotowire
IDENTIFIER=final_model
GPUID=0

python train.py -data $BASE/preprocess/roto-encdec-history-window-3 -save_model $BASE/gen_model/$IDENTIFIER/roto -two_dim_score mlp  -hier_history_seq_type SA -hier_hist_attn_type mlp -hier_hist_attn_pos_type posEmb -hier_history_seq_window 3 -hier_meta $BASE/hier_meta.json -hier_rnn_size 600 -hier_bidirectional -encoder_type1 mean -enc_layers1 2 -encoder_type2 mean -decoder_type2 rnn -dec_layers2 2 -batch_size 5 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -start_checkpoint_at 4 -epochs 50 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid $GPUID -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size 5
```

## translate

The following command will generate on test set given trained model. Results on valid set can be obtained by changing -src $BASE/inf_src_valid.txt -src_hist $BASE/hist_full/inf_src_valid_hist_3.txt and -output.

```
BASE=./rotowire
GPUID=3

python translate.py -model2 ./paper_model/model.pt -src $BASE/inf_src_test.txt -src_hist $BASE/hist_full/inf_src_test_hist_3.txt -output ./paper_model/model_test.txt -batch_size 10 -max_length 850 -gpu $GPUID -min_length 150
```

## evaluation

The following command will produce BLEU metric of model's generation on test set. The evaluation script can be obtained from link https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl

```
perl ref/multi-bleu.perl ref/test.txt < ./paper_model/model_test.txt
```

As for obtaining extractive evaluation metrics, please refer to https://github.com/ratishsp/data2text-1 for details.