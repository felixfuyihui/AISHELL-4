#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

dataset="aishell_v4" # prepare data in data/aishell_v4/{train,dev,test}
# training
gpu=4 #gpuid
am_exp=asr # load training configuration in conf/aishell_v4/asr.yaml

seed=777
tensorboard=false
prog_interval=200

# for am
am_epochs=50 
am_batch_size=24 
am_num_workers=16

. ./utils/parse_options.sh || exit 1

./scripts/train.sh \
  --seed $seed \
  --gpu $gpu \
  --epochs $am_epochs \
  --batch-size $am_batch_size \
  --num-workers $am_num_workers \
  --tensorboard $tensorboard \
  --prog-interval $prog_interval \
  am $dataset $am_exp

