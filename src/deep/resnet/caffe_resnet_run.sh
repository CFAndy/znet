#!/bin/bash

cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
znet_path="${cur_dir}/../"
# echo ${znet_path}
caffe_path="${cur_dir}/../../../caffe/python"
# echo ${caffe_path}

export PYTHONPATH=${znet_path}:${caffe_path}:$PYTHONPATH
echo $PYTHONPATH

# export GLOG_minloglevel=2

unset OMP_NUM_THREADS
# export OMP_NUM_THREADS=44
# export MKL_NUM_THREADS

# export OMP_WAIT_POLICY=passive
unset MKL_THREADING_LAYER
# export MKL_THREADING_LAYER=gnu
unset KMP_AFFINITY
# export KMP_AFFINITY=compact,1,0,granularity=fine

solver_proto="${cur_dir}/../../../models/wrn/solver.prototxt"
../../../caffe/build/tools/caffe train --solver=${solver_proto}
