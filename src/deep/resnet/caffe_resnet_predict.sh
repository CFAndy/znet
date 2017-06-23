#!/bin/bash

cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
znet_path="${cur_dir}/../"
# echo ${znet_path}
caffe_path="${cur_dir}/../../../caffe/python"
# echo ${caffe_path}

export PYTHONPATH=${znet_path}:${caffe_path}:$PYTHONPATH
echo $PYTHONPATH

export GLOG_minloglevel=2

core_num=`nproc`

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=${core_num}
export MKL_NUM_THREADS=${core_num}
# export OMP_DYNAMIC="False"

# export OMP_WAIT_POLICY=passive
unset MKL_THREADING_LAYER
# export MKL_THREADING_LAYER=gnu
export KMP_AFFINITY=compact,1,0,granularity=fine

model_path="${cur_dir}/../../../snapshots/wrn_iter_14000.caffemodel"
echo ${model_path}
proto_path="${cur_dir}/../../../models/wrn/wide_resnet_prediction.prototxt"
echo ${proto_path}

python resnet_prediction.py ${model_path} ${proto_path}
