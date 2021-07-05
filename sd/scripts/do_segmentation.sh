. ./path.sh
. ./cmd.sh

# local/segmentation/detect_speech_activity.sh --nj 4 --stage 0 --cmd run.pl \
#     /home/lycheng/workspace/corecode/Python/kaldi_chime6_sad/0012_sad_v1/corpus/test_0328/ \
#     exp/segmentation_1a/tdnn_stats_sad_1a/ \
#     /home/lycheng/workspace/corecode/Python/kaldi_chime6_sad/0012_sad_v1/corpus/test_0328/mfcc/ \
#     sad_work/test_0328/ \
#     /home/lycheng/workspace/corecode/Python/kaldi_chime6_sad/0012_sad_v1/corpus/test_0328/sad/

data_dir=$1
sad_work=$2
sad_result=$3
nj=$4

echo "$data_dir"
echo "$sad_work"
echo "$sad_result"
echo "$nj"

local/segmentation/detect_speech_activity.sh --nj $nj --stage 0 --cmd run.pl \
    $data_dir exp/segmentation_1a/tdnn_stats_sad_1a/ $data_dir/feat/mfcc $sad_work $sad_result

