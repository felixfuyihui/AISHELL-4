#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu


dataset="aishell_v4" # prepare data in data/aishell_v1/{train,dev,test}

gpu=4 #gpuid
am_exp=asr # load configuration in conf/aishell_v4/asr.yaml

seed=777
tensorboard=false
prog_interval=200




# decoding
beam_size=24
nbest=8
. ./utils/parse_options.sh || exit 1

decode_wav_scp=$1
dec_dir=$2


beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

# decoding
./scripts/decode.sh \
  --score true \
  --gpu $gpu \
  --beam-size $beam_size \
  --nbest $nbest \
  --max-len 50 \
  --dict data/$dataset/dict \
  $dataset $am_exp \
  $decode_wav_scp \
  exp/$dataset/$am_exp/${dec_dir} \
