
# The tools need you to change the kaldi root by yourself
export KALDI_ROOT=/home/work_nfs/common/kaldi-20190604-cuda10

work_path=$1
wav_path=$2
exp_path=$work_path/exp

mkdir -p $exp_path || exit 1;

for audio in $(awk '{print $1}' $work_path/wav.scp)
do
    filename="${audio}"
    echo $filename > $exp_path/${filename}_wav_list.txt
    echo "Sub jobs predict for $filename"
    $train_cmd $work_path/exp/extract_embedding_${filename}.log \
    python VBx/predict.py --in-file-list $exp_path/${filename}_wav_list.txt \
                          --in-lab-dir $work_path/vad \
                          --in-wav-dir $wav_path \
                          --out-ark-fn $work_path/embedding/${audio}.ark \
                          --out-seg-fn $work_path/embedding/${audio}.seg \
                          --weights VBx/models/ResNet101_16kHz/nnet/final.onnx \
                          --backend onnx &
    echo "${filename} finished"
done
echo "extract embedding finished!"

