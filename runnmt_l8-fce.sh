#!/bin/bash

#THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn,device=gpu0 python nematus/nmt.py \
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn,device=gpu python nematus/nmt.py \
  --datasets data/corpus/train.l8-fce.src data/corpus/train.l8-fce.trg \
  --dictionaries data/corpus/train.l8-fce.src.json data/corpus/train.l8-fce.trg.json \
  --model models/mle-l8-fce.npz \
  --use_dropout \
  --maxlen 50 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 40 \
  --n_words_src 35000 \
  --n_words 35000
