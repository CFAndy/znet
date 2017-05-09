#!/bin/bash

cur_dir=$(cd "$(dirname $0)";pwd)

candidates="finalizedcandidates_unet_ALL.csv"
python create_xy_xz_yz_CARTESIUS.py 0 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 1 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 2 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 3 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 4 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 5 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 6 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 7 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 8 ${candidates}
python create_xy_xz_yz_CARTESIUS.py 9 ${candidates}
