echo i$1

export PYTHONPATH=$PYTHONPATH:/workshop/matrix/znet/src/deep


/home/2T/intel-caffe/build/tools/caffe train --solver=$1 --iterations=1
