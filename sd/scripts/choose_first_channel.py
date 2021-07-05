import argparse
import tqdm
import os

import soundfile as sf


def main(args):
    init_wav_scp_file = args.init_wav_scp_file
    output_wav_path = args.output_wav_path
    output_wav_scp_file = args.output_wav_scp_file

    input_wav_scp = dict()
    with open(init_wav_scp_file, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            parts = line.strip().split()
            input_wav_scp[parts[0]] = parts[1]

    output_wav_scp = dict()
    for utt_id in tqdm.tqdm(input_wav_scp.keys()):
        input_utt_path = input_wav_scp[utt_id]
        utt, sr = sf.read(input_utt_path)
        print(utt.shape)
        print(utt.T[0].shape)

        output_utt_path = os.path.join(output_wav_path, utt_id + ".wav")
        sf.write(output_utt_path, utt.T[0], sr)
        output_wav_scp[utt_id] = output_utt_path

    with open(output_wav_scp_file, "w") as fw:
        for (utt_id, utt_path) in output_wav_scp.items():
            fw.write(f"{utt_id} {utt_path}\n")
        fw.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Choose one channel from multi-channel wav")
    parser.add_argument("--init_wav_scp_file", required=True, help="The init wav.scp file")
    parser.add_argument("--output_wav_path", required=True, help="The output wav path")
    parser.add_argument("--output_wav_scp_file", required=True, help="The output wav.scp file")
    args = parser.parse_args()
    main(args)
