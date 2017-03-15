import numpy as np
from dataset_2D import load_images
from functools import partial
from parallel import ParallelBatchIterator
from augment import testtime_augmentation
import glob
import pandas as pd
from params import wrn_params as P
import caffe
import os
import sys

def get_images_with_filenames(filenames):
    ####read a mini batch data
    inputs, targets = load_images(filenames, deterministic=True)
    new_inputs = []
    new_targets = []
    for image, target in zip(inputs, targets):
        ims, trs = testtime_augmentation(image[0], target) #Take color channel of image
        new_inputs += ims
        new_targets+=trs
    new_filenames = []
    for fname in filenames:
        for i in range(int(len(new_inputs)/len(filenames))):
            new_filenames.append(fname)
    return np.array(new_inputs,dtype=np.float32),np.array(new_targets,dtype=np.int32), new_filenames


def main():
    model_path = sys.argv[1]
    net_file = sys.argv[2]
    ###laod all predicting filenames
    filenames = glob.glob(P.FILENAMES_PREDICTION)
    batch_size = P.BATCH_SIZE_PREDICTION
    ###get augmentation number
    test_im = np.zeros((64,64))
    n_testtime_augmentation = len(testtime_augmentation(test_im, 0)[0])
    ####get the parallel data generator
    gen = ParallelBatchIterator(get_images_with_filenames,
            filenames, ordered=True,
            batch_size=batch_size//(3*n_testtime_augmentation),
            multiprocess=False, n_producers=12)
    ###get wide resnet caffe model
    caffe_net = caffe.Net(net_file, model_path, caffe.TEST)
    ###do forward pass
    all_probabilities = []
    all_filenames = []
    print('begin predicting...')
    for i, batch in enumerate(gen):
        data, label, fnames =  batch
        ###pass data and label to caff net, please set the batchszie to 12(atomic batchsize) for both prediction batch size and train batch size
        if data.shape[0] == batch_size:
            caffe_net.blobs['data'].data[...] = data.astype(np.float32, copy=False)
            caffe_net.blobs['label'].data[...] = label.astype(np.float32, copy=False)
            caffe_net.forward()
            softmax_out = caffe_net.blobs['prob'].data.copy()
            all_probabilities += list(softmax_out[:, 1].tolist())
            all_filenames += list(fnames)
        else:
            break
    ###for all filenames get the probalilities(3*n_testtime_augmentation)
    d = {f:[] for f in filenames}
    for probability, f in zip(all_probabilities, all_filenames):
        d[f].append(probability)
    ###the code will no run unless the batch size is not the atomic batch size(3*n_testtime_augmentation)
    for key in d.keys():
        if len(d[key]) == 0:
            d.pop(key)
    print('end predicting')
    print('write to csv')
    candidates = pd.read_csv('../../data/candidates_V2.csv')
    data = []
    ###get the mean prob for each filename and get the row number of the candidate
    for x in d.iteritems():
        fname, probabilities = x
        prob = np.mean(probabilities)
        candidates_row = int(os.path.split(fname)[1].replace('.pkl.gz','')) - 2
        data.append(list(candidates.iloc[candidates_row].values)[:-1]+[str(prob)])
    ###write the prob to a .csv
    submission = pd.DataFrame(columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'], data=data)
    submission_path = '../../data/submission_subset45.csv'
    submission.to_csv(submission_path, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
    print('finished!')

if __name__ == '__main__':
    main()

