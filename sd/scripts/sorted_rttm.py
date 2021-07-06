import argparse
import os


def read_rttm(rttm_file):
    rttm = dict()
    with open(rttm_file, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            parts = line.strip().split()
            utt_id, start, duration, speaker_id = parts[1], float(parts[3]), float(parts[4]), parts[7]
            # print(utt_id, " ", start, " ", duration, " ", speaker_id)
            if utt_id not in rttm:
                rttm[utt_id] = dict()
            if speaker_id not in rttm[utt_id]:
                rttm[utt_id][speaker_id] = list()
            rttm[utt_id][speaker_id].append((start, duration))
    return rttm


def write_rttm_file(filename, rttm):
    rttm = sorted(rttm, key=lambda x:x[2])
    with open(filename, "w") as fw:
        for (utt_id, spk_id, start, end) in rttm:
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            fw.write(f"{fmt.format(utt_id, start, end - start, spk_id)}\n")


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    input_rttm_list = os.listdir(input_dir)

    for input_rttm_file in input_rttm_list:
        cur_rttm_file = os.path.join(input_dir, input_rttm_file)
        cur_rttm = read_rttm(cur_rttm_file)
        result_rttm = list()
        result_rttm_file = os.path.join(output_dir, input_rttm_file)
        print(input_rttm_file)

        for index, utt_id in enumerate(cur_rttm.keys()):
            cur_spk_dict = cur_rttm[utt_id]
            cur_spk2dur = dict()
            for spk_id in cur_spk_dict.keys():
                if spk_id not in cur_spk2dur:
                    cur_spk2dur[spk_id] = 0.0
                for (onset, offset) in cur_spk_dict[spk_id]:
                    cur_spk2dur[spk_id] += offset
            spk2dur = list(cur_spk2dur.items())
            spk2dur = sorted(spk2dur, key=lambda x: x[1])
            spk_map = {spk_id: str(index + 1) for index, (spk_id, _) in enumerate(spk2dur)}
            print(utt_id)
            print(spk2dur)
            print(spk_map)

            for spk_id in cur_spk_dict.keys():
                for (onset, offset) in cur_spk_dict[spk_id]:
                    result_rttm.append((utt_id, spk_map[spk_id], onset, offset + onset))
        write_rttm_file(result_rttm_file, result_rttm)
        print("write to rttm file " + result_rttm_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("For stable rttm, sort the spk id by dur")
    parser.add_argument("--input_dir", required=True, help="The input dir, you should make sure the one rttm have only one utterances")
    parser.add_argument("--output_dir", required=True, help="The output dir, the same filename as the input dir")
    args = parser.parse_args()
    main(args)
