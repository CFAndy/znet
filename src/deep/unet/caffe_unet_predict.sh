#!/bin/bash

cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
znet_path="${cur_dir}/../"
# echo ${znet_path}
caffe_path="${cur_dir}/../../../caffe/python"
# echo ${caffe_path}

export PYTHONPATH=${znet_path}:${caffe_path}:$PYTHONPATH
echo $PYTHONPATH

snapshot="${cur_dir}/../../../snapshots/unet_iter_311061.caffemodel"
prototxt="${cur_dir}/../../../models/unet/unet_prediction.prototxt"
python ./predict.py  ${snapshot} ${prototxt}
