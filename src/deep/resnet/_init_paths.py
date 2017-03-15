"""Set up Python paths."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.abspath(__file__))

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '../../..', 'caffe', 'python')
# print caffe_path
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '../')
# print lib_path
add_path(lib_path)
