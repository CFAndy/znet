#!/bin/bash

cur_dir=$(cd "$(dirname $0)";pwd)
# echo ${cur_dir}
model_path="${cur_dir}/../../../wrn_weight/_iter_9000.caffemodel"
# echo ${model_path}
proto_path="${cur_dir}/../../../models/wrn/wide_resnet_prediction.prototxt"
# echo ${proto_Path}

python resnet_prediction.py ${model_path} ${proto_path}
