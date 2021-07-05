
aishell4_test_dir=$1
work_path=$2

mkdir -p $work_path/one_channel_wav || exit 1;

python scripts/choose_first_channel.py --init_wav_scp_file $work_path/init_wav.scp \
                                       --output_wav_path $work_path/one_channel_wav/ \
                                       --output_wav_scp_file $work_path/wav.scp

awk '{print $1" "$1}' $work_path/wav.scp > $work_path/utt2spk
awk '{print $1" "$1}' $work_path/wav.scp > $work_path/spk2utt

