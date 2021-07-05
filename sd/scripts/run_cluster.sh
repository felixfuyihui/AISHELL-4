
. path.sh

work_path=$1
exp_path=$work_path/exp

for audio in $(awk '{print $1}' $work_path/wav.scp);
do
    filename="${audio}"
    echo $filename
    $train_cmd $exp_path/cluster_${filename}.log \
        python VBx/vbhmm.py --init AHC+VB \
                --out-rttm-dir $work_path/rttm \
                --xvec-ark-file $work_path/embedding/${audio}.ark \
                --segments-file $work_path/embedding/${audio}.seg \
                --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
                --plda-file VBx/models/ResNet101_16kHz/plda \
                --threshold -0.015 \
                --lda-dim 128 \
                --Fa 0.3 \
                --Fb 17 \
                --loopP 0.99 &
    echo "$filename mission added"
done

