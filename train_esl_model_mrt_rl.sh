#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn,device=gpu python nematus/nmt.py \
  --datasets ./data/corpus/train-esl.src ./data/corpus/train-esl.trg \
  --dictionaries ./data/corpus/train-esl.src.json ./data/corpus/train-esl.trg.json \
  --model ./models/mrt-rl/esl.npz \
  --reload \
  --objective MRT \
  --mrt_samples 20 \
  --mrt_samples_meanloss 20 \
  --mrt_reference \
  --saveFreq 20000 \
  --finish_after 1000000 \
  --use_dropout \
  --maxlen 50 \
  --optimizer sgd \
  --lrate 0.0001 \
  --n_words_src 35000 \
  --n_words 35000
