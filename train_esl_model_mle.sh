#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn,device=gpu python nematus/nmt.py \
  --datasets ./data/corpus/train-esl.src ./data/corpus/train-esl.trg \
  --dictionaries ./data/corpus/train-esl.src.json ./data/corpus/train-esl.trg.json \
  --model models/mle/esl.npz \
  --reload \
  --saveFreq 20000 \
  --use_dropout \
  --maxlen 50 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 40 \
  --n_words_src 35000 \
  --n_words 35000
