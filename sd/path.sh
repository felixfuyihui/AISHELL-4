export KALDI_ROOT=/home/work_nfs/common/kaldi-20190604-cuda10
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

SDT_ROOT=`pwd`
export PYTHONPATH=$SDT_ROOT:$PYTHONPATH
export PYTHONPATH=$SDT_ROOT/local:$PYTHONPATH
export PYTHONPATH=$SDT_ROOT/scripts:$PYTHONPATH

