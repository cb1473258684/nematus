#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,optimizer_including=cudnn,device=gpu python nematus/nmt.py \
  --datasets ./data/corpus/train-esl.src ./data/corpus/train-esl.trg \
  --dictionaries ./data/corpus/train-esl.src.json ./data/corpus/train-esl.trg.json \
  --model ./models/mrt_mle_0/esl.npz \
  --objective MRT \
  --mrt_reference \
  --mrt_samples 20 \
  --mrt_ml_mix 1 \
  --saveFreq 20000 \
  --use_dropout \
  --maxlen 50 \
  --optimizer adam \
  --lrate 0.0001 \
  --batch_size 40 \
  --n_words_src 35000 \
  --n_words 35000
