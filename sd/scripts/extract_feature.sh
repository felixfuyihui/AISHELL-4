. ./path.sh
. ./cmd.sh

dataset=$1
mfccdir=$2
nj=$3

steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
    --mfcc-config conf/mfcc_hires.conf \
    $dataset $dataset/make_mfcc $mfccdir

