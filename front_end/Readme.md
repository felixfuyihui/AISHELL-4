# front_end

1. Please prepare the `mix.scp`, `spk1.scp`, `spk2.scp`, `utt2dur` for training and development set under `data/train/` and `data/dev/` respectively following the kaldi format.

2. Please prepare the `mix_nospk.scp` and `mix_spk.scp` for testing data of Speaker Independent and Speaker Dependent tasks under `data/test/` respectively following the kaldi format.

3. Train the fe model
```bash
./run.sh
```

4. Generate the fe results for Speaker Independent and Speaker Dependent tasks respectively
```bash
./decode.sh data/test/wav_nospk.scp
./decode.sh data/test/wav_spk.scp
```
You can also use our pretrained model to generate the fe results directly. Please download the pretrain the model from xxx and save at `exp/realmask/`.
