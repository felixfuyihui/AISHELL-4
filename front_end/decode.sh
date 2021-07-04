#/bin/bash

set -euo pipefail

lr="1e-3"

encoder_norm_type='cLN'
save_name="realmask"
mkdir -p exp/${save_name}

num_gpu=1
batch_size=$[num_gpu*1]
wav_scp_test=$1

CUDA_VISIBLE_DEVICES="4" \
python -u steps/run_realmask.py \
--decode="true" \
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
--mix-scp-dev="data/train/mix.scp" \
--s1-scp-dev="data/train/spk1.scp" \
--s2-scp-dev="data/train/spk2.scp" \
--utt2dur-dev="data/train/utt2dur" \
--wav-scp-test=${wav_scp_test}