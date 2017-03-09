#!/bin/bash

# translate
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu1 python nematus/translate.py --models models/mle-l8-fce-giga.npz --input data/corpus/dev.src > hyp/hyp.tmp

# recaser

# capitalize the first char (if necessary)

# copy src sentences when longer than 50 words (do nothing)

# run GLEU.
python2 ../jfleg/eval/gleu.py -r ../jfleg/dev/dev.ref[0-3] -s ../jfleg/dev/dev.src --hyp ./hyp/hyp.tmp
