1. Please prepare the `wav.scp`, `text`, `utt2dur` for training and development set under `data/aishell_v4/train/` and `data/aishell_v4/dev/` respectively following the kaldi format.
   Note: Both training and development data contain two parts: original clean speech to simulate data and simulated noisy speech. Please combine these two parts together to train the asr model.

2. Please prepare the `wav_nospk_nofe.scp`, `wav_spk_nofe.scp`, `wav_nospk_fe.scp` and `wav_spk_fe.scp` for testing data of Speaker Independent and Speaker Dependent tasks under `data/aishell_v4/test/` respectively following the kaldi format.

3. Train the asr model
```bash
./run.sh
```

4. Generate the asr results
```bash
./decode.sh data/test/wav_nospk_nofe.scp wav_nospk_nofe
./decode.sh data/test/wav_spk_nofe.scp wav_spk_nofe
./decode.sh data/test/wav_nospk_fe.scp wav_nospk_fe
./decode.sh data/test/wav_spk_fe.scp wav_spk_fe
```
You can also use our pretrained model to generate the asr results directly. Please download the pretrain the model from xxx and save at `exp/aishell_v4/asr/`.
