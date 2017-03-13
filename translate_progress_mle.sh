#!/bin/bash

# make sure you have run ./models_old/copy_npz_jsons.sh in advance!

for i in $(seq 20 20 300) ; do # seq from diff upto
  THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python ./nematus/translate.py --models ./models/mle/esl.iter${i}000.npz --input ./data/corpus/dev.src > ./hyp/mle_iter.${i}k.hyp
done
