#/bin/bash

set -euo pipefail

lr="1e-3"

encoder_norm_type='cLN'
save_name="release_estimask" # experiment name
mkdir -p exp/${save_name}

num_gpu=8
batch_size=$[num_gpu*84] #12

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -u steps/run_realmask.py \
--decode="false" \
--batch-size=${batch_size} \
--learning-rate=${lr} \
--weight-decay=1e-5 \
--epochs=20 \
--use-cuda="true" \
--model-dir="exp/${save_name}" \
--mix-scp-tr="data/train/mix.scp" \
--s1-scp-tr="data/train/spk1.scp" \
--s2-scp-tr="data/train/spk2.scp" \
--utt2dur-tr="data/train/utt2dur" \
--mix-scp-dev="data/dev/mix.scp" \
--s1-scp-dev="data/dev/spk1.scp" \
--s2-scp-dev="data/dev/spk2.scp" \
--utt2dur-dev="data/dev/utt2dur"
