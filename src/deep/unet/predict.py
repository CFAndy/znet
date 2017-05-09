from __future__ import division
import sys
import numpy as np
import os
import skimage.io
import caffe
from utils import output_size_for_input
from udataset import load_images
from parallel import ParallelBatchIterator
from functools import partial
from tqdm import tqdm
from glob import glob
import util

from params import unet_prediction_params as P

P.RANDOM_CROP = 0
NET_DEPTH = P.DEPTH # Default 5
INPUT_SIZE = P.INPUT_SIZE
OUTPUT_SIZE = output_size_for_input(INPUT_SIZE, NET_DEPTH)

if __name__ == "__main__":
    if len(sys.argv) < 3:
      print "first parameter is model file, second parameter is net file"
      quit()

    model_file = sys.argv[1]
    net_file = sys.argv[2]

    ### load all predicting filenames
    file_names = glob(P.FILENAMES_PREDICTION)
    batch_size = P.BATCH_SIZE_PREDICTION

    multiprocess = False
    gen = ParallelBatchIterator(partial(load_images, deterministic = True),
                                        file_names, ordered = True,
                                        batch_size = batch_size,
                                        multiprocess = multiprocess)

    caffe_net = caffe.Net(net_file, model_file, caffe.TEST)

    predictions_folder = model_file + '_predictions'
    util.make_dir_if_not_present(predictions_folder)

    all_probabilities = []
    all_filenames = []
    for i, batch in enumerate(tqdm(gen)):
        inputs, labels, weights, fnames = batch

        if inputs.shape[0] == batch_size:
            caffe_net.blobs['data'].data[...] = inputs.astype(np.float32, copy = False)
            caffe_net.forward()
            softmax_out = caffe_net.blobs['prob'].data.copy()
            all_probabilities += list(softmax_out[:, 1, :, :].tolist())
            all_filenames += list(fnames)
            # print("one batch done")
        else:
            print("data shape[0] is not equal to batch size, break out")
            break

        # print inputs.shape, weights.shape
        for n, filename in enumerate(fnames):
            # Whole filepath without extension
            f = os.path.splitext(os.path.splitext(filename)[0])[0]

            # Filename only
            subdir = os.path.basename(os.path.dirname(f))
            f = os.path.basename(f)
            sub_folder = os.path.join(predictions_folder, subdir)
            util.make_dir_if_not_present(sub_folder)
            f = os.path.join(sub_folder, f + '.png')
            out_size = output_size_for_input(inputs.shape[3], NET_DEPTH)
            image_size = out_size ** 2
            image = softmax_out[n, 1, :, :].reshape(out_size, out_size)

            # Remove parts outside a few pixels from the lungs
            image = image * np.where(weights[n, 0, :, :] == 0, 0, 1)

            image = np.array(np.round(image * 255), dtype = np.uint8)

            skimage.io.imsave(f, image)
