export PYTHONPATH=/home/matrix/znet/src/deep:/home/matrix/py-R-FCN-MKL/caffe/python:$PYTHONPATH

export GLOG_minloglevel=2

unset OMP_NUM_THREADS
# export OMP_NUM_THREADS=44
# export MKL_NUM_THREADS

# export OMP_WAIT_POLICY=passive
unset MKL_THREADING_LAYER
# export MKL_THREADING_LAYER=gnu

/home/matrix/py-R-FCN-MKL/caffe/build/tools/caffe train --solver=/home/matrix/znet/models/unet/unet_solver.prototxt --iterations=1
