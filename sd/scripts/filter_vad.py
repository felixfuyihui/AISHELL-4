import argparse


def merge(segment_list, tolerance_interval: float):
    result_list = list()
    start_time = -1.0
    end_time = -1.0
    for (start, end) in segment_list:
        if start - end_time > tolerance_interval:
            if start_time != -1.0:
                result_list.append((start_time, end_time))
            start_time = start
            end_time = end
        else:
            end_time = end
    result_list.append((start_time,end_time))
    return result_list


def filter_short(cur_list, filter_short_factor):
    return [(start, end) for (start, end) in cur_list if end - start > filter_short_factor]


def main(args):
    input_segments = args.input_segments
    short_utterance = args.short_utterance
    short_interval = args.short_interval
    output_segments = args.output_segments

    def read_segments(input_segments_file):
        segments_list = list()
        with open(input_segments_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                parts = line.strip().split()
                segments_list.append((
                    parts[0], parts[1], float(parts[2]), float(parts[3])
                ))
        return segments_list

    segments_list = read_segments(input_segments)
    utt2segments = dict()
    for (segment_utt_id, utt_id, start, end) in segments_list:
        if utt_id not in utt2segments:
            utt2segments[utt_id] = []
        utt2segments[utt_id].append((start, end))

    result_segments_list = []
    for utt_id in utt2segments.keys():
        print(utt_id)
        segment_list = utt2segments[utt_id]
        segment_list.sort()
        print(len(segment_list))
        segment_list = merge(segment_list, short_interval)
        print(len(segment_list))
        segment_list = filter_short(segment_list, short_utterance)
        print(len(segment_list))

        for index, (start, end) in enumerate(segment_list):
            result_segments_list.append((
                f"{utt_id}_{str(index).zfill(5)}", utt_id, start, end
            ))

    print(f"Collect {len(result_segments_list)} utterances")
    def write_segments(segments_file, segments_list):
        with open(segments_file, "w") as fw:
            for (segment_utt_id, utt_id, start, end) in segments_list:
                fw.write(f"{segment_utt_id} {utt_id} {start} {end}\n")
    write_segments(output_segments, result_segments_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Filter segments: filter short and merge short intervals")
    parser.add_argument("--input_segments", required=True, help="The input segments")
    parser.add_argument("--short_utterance", default=1., type=float, help="The short utterance")
    parser.add_argument("--short_interval", default=1., type=float, help="The short interval")
    parser.add_argument("--output_segments", required=True, help="The output segments")

    args = parser.parse_args()
    main(args)
