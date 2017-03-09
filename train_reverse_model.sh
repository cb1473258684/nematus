#!/bin/bash

# train from trg to src! 
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn,device=gpu0 python nematus/nmt.py \
  --datasets ./data/corpus/train-esl.trg ./data/corpus/train-esl.src \
  --dictionaries ./data/corpus/train-esl.trg.json ./data/corpus/train-esl.src.json \
  --valid_datasets ./data/corpus/dev.trg ./data/corpus/dev.src \
  --model models/reverse.npz \
  --use_dropout \
  --maxlen 50 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 40 \
  --n_words_src 35000 \
  --n_words 35000
