import argparse
import os
import typing


def parse_rttm(segment_list: typing.List, max_speech_length: float = 9.5) -> typing.List:
    result_segment_list = list()
    for (utt_id, spk_id, start, duration) in segment_list:
        while duration > max_speech_length + 1.0:
            result_segment_list.append((utt_id, spk_id, start, max_speech_length))
            duration -= max_speech_length
            start += max_speech_length
        result_segment_list.append((utt_id, spk_id, start, duration))
    return result_segment_list


def read_rttm(rttm_file) -> typing.List:
    segment_list = list()
    with open(rttm_file, "r") as fr:
        lines = fr.readlines()
        for line in lines:
            parts = line.strip().split()
            segment_list.append((parts[1], parts[7], float(parts[3]), float(parts[4])))
    return segment_list


def write_rttm(output_rttm_file, segment_list):
    with open(output_rttm_file, "w") as fw:
        for (infer_id, spk_id, start, end) in segment_list:
            fmt = "SPEAKER {:s} 1 {:7.5f} {:7.5f} <NA> <NA> {:s} <NA> <NA>"
            fw.write(f"{fmt.format(infer_id, start, end, spk_id)}\n")


def main(args):
    init_path = args.init_path
    result_path = args.result_path

    items = os.listdir(init_path)
    for item in items:
        print(item)
        rttm_file = os.path.join(init_path, item)
        segment_list = read_rttm(rttm_file)
        segment_list = parse_rttm(segment_list)
        write_rttm(os.path.join(result_path, item), segment_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Split rttm")
    parser.add_argument("--init_path", required=True, help="The initial path for old rttm")
    parser.add_argument("--result_path", required=True, help="The result_path to save the rttm")
    args = parser.parse_args()
    main(args)

