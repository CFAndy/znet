echo i$1
export PYTHONPAT=$PYTHONPAT:/home/2T/luna16/src/deep/lib
#python test.py
#/home/2T/intel-caffe/build/tools/caffe time --model=$1 --iterations=5
/home/2T/intel-caffe/build/tools/caffe train --solver=$1 --iterations=1
#rm _iter_*.* -rf
