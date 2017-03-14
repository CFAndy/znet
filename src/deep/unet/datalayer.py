import caffe
import numpy as np
import sys
import glob

sys.path.append('../')
from params import params as P
import dataset

def output_size_for_input(in_size, depth):
    for _ in range(depth - 1):
        in_size = in_size // 2
    for _ in range(depth - 1):
        in_size = in_size * 2
    return in_size

NET_DEPTH = P.DEPTH # Default 5
INPUT_SIZE = P.INPUT_SIZE # Default 512
OUTPUT_SIZE = output_size_for_input(INPUT_SIZE, NET_DEPTH)

class DataLayer(caffe.Layer):
    def read_data(self):
        l, t, w, _ = dataset.load_images(self.train_data[self.index : self.index + self.batch_size])
        self.index += self.batch_size
        if self.index + self.batch_size > len(self.train_data):
            self.index = 0
            np.random.shuffle(self.train_data)
        return l, t

    def setup(self, bottom, top):
        # print ("setup")
        # for debug
        np.random.seed(0)
        # print P.FILENAMES_TRAIN
        file_names = glob.glob(P.FILENAMES_TRAIN)        
        train_splits = dataset.train_splits_by_z(file_names, 0.3, P.N_EPOCHS)
        print "train number per epoch: ", len(train_splits[0])
        self.train_data = [item for sublist in train_splits for item in sublist]
        print "total train number: ", len(self.train_data)

        self.index = 0
        self.batch_size = P.BATCH_SIZE_TRAIN
        # sys.exit(0)

        idx = 0
        top[idx].reshape(self.batch_size, P.CHANNELS, INPUT_SIZE, INPUT_SIZE)
        idx += 1
        top[idx].reshape(self.batch_size, 1, OUTPUT_SIZE, OUTPUT_SIZE)

    def forward(self, bottom, top):
        # print "datalayer start forward"
        data, label = self.read_data()
        if label.shape[0] != self.batch_size:
            top[0].reshape(data.shape[0], P.CHANNELS, INPUT_SIZE, INPUT_SIZE)
            top[1].reshape(label.shape[0], 1, OUTPUT_SIZE, OUTPUT_SIZE)
            print ('reshape ', label.shape[0])
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
        else:
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
        # print ("read data\n")
        # sys.exit(0)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class ValDataLayer(caffe.Layer):
    def read_data(self):
        l, t, w, _ = dataset.load_images(self.data[self.index:self.index + self.batch_size])
        self.index += self.batch_size
        if self.index + self.batch_size > len(self.data):
            self.index = 0
            np.random.shuffle(self.data)
        return l, t

    def setup(self, bottom, top):
        # print ("setup")
        # for debug
        np.random.seed(0)
        self.data = glob.glob(P.FILENAMES_VALIDATION)        
        self.batch_size = P.BATCH_SIZE_VALIDATION

        # print len(self.data)
        self.index = 0
        # sys.exit(0)

        idx = 0
        top[idx].reshape(self.batch_size, P.CHANNELS, INPUT_SIZE, INPUT_SIZE)
        idx += 1
        top[idx].reshape(self.batch_size, 1, OUTPUT_SIZE, OUTPUT_SIZE)

    def forward(self, bottom, top):
        data, label = self.read_data()
        if label.shape[0] != self.batch_size:
            top[0].reshape(data.shape[0], P.CHANNELS, INPUT_SIZE, INPUT_SIZE)
            top[1].reshape(label.shape[0], 1, OUTPUT_SIZE, OUTPUT_SIZE)
            print ('reshape ', label.shape[0])
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
        else:
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
        # print ("read data\n")
        # sys.exit(0)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
