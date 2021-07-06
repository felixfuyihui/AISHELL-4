import argparse
import os


def main(args):
    init_wav_path = args.init_wav_path
    init_wav_scp_file = args.init_wav_scp_file

    with open(init_wav_scp_file, "w") as fw:
        for item in os.listdir(init_wav_path):
            print(item)
            if not str(item).endswith(".wav"):
                continue
            fw.write(f"{item[:-4]} {os.path.join(init_wav_path, item)}\n")

    print("Prepare init wav.scp finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Make aishell4 test")
    parser.add_argument("--init_wav_path", required=True, help="The init wav path")
    parser.add_argument("--init_wav_scp_file", required=True, help="The init wav.scp file")
    args = parser.parse_args()
    main(args)
