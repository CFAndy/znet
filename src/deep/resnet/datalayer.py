import caffe
import numpy as np
import sys
from params import params as P
import glob
import dataset_2D
#from parallel import ParallelBatchIterator

n_channels = P.CHANNELS
PIXELS = P.INPUT_SIZE
batch_size = P.BATCH_SIZE_TRAIN // 3
W=1
class DataLayer(caffe.Layer):
    def _shuffle_data(self):
        np.random.shuffle(self.file_false)
        self.epoch_data = self.file_true + self.file_false[:self.n_true]
        np.random.shuffle(self.epoch_data)

    def read_data(self):
        #np.random.shuffle(self.file_false)
        self.epoch_data = self.file_true +  self.file_false[:self.n_true]
        #np.random.shuffle(self.epoch_data)
        l,t,= dataset_2D.load_images(self.epoch_data[self.index:self.index+batch_size])
        #print (l.shape,t.shape,batch_size,len(self.epoch_data))
        self.index +=batch_size
        if self.index + batch_size > len(self.epoch_data):
            self.index = 0
            self._shuffle_data()
        return l,t

    def _yield_data(self):
        while True:
            for batch in self.batch_generator:
                yield batch
            print('done')
            self._shuffle_data()

    def setup(self, bottom, top):
        #print ("set up")
        file_names = glob.glob(P.FILENAMES_TRAIN)
        self.file_true = filter(lambda x: "True" in x, file_names)
        self.file_false = filter(lambda x: "False" in x, file_names)
        self.n_true = len(self.file_true)
        self._shuffle_data()
        self.data_generator = self._yield_data()
        self.index=0
        #sys.exit(0)

        idx = 0
        top[idx].reshape(P.BATCH_SIZE_TRAIN, P.CHANNELS, PIXELS, PIXELS)
        idx += 1
        top[idx].reshape(P.BATCH_SIZE_TRAIN, W)

    def forward(self, bottom, top):
        #data, label = self.data_generator.next()
        data, label = self.read_data()
        if label.shape[0] != P.BATCH_SIZE_TRAIN:
            top[0].reshape(data.shape[0],P.CHANNELS, PIXELS, PIXELS)
            top[1].reshape(label.shape[0],W)
            print ('reshape ', label.shape[0])
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
        else:
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
        #print ("read data\n")
        #sys.exit(0)
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class ValDataLayer(caffe.Layer):
    def _shuffle_data(self):
        np.random.shuffle(self.file_false)
        self.epoch_data = self.file_true + self.file_false[:self.n_true]
        np.random.shuffle(self.epoch_data)
        #self.batch_generator = ParallelBatchIterator(dataset_2D.load_images, self.epoch_data, ordered=False, batch_size=P.BATCH_SIZE_TRAIN//3,
        #        multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION, n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

    def read_data(self):
        self.epoch_data = self.file_true +  self.file_false[:self.n_true]
        l,t,= dataset_2D.load_images(self.epoch_data[self.index:self.index+batch_size],True)
        self.index +=batch_size
        if self.index + batch_size > len(self.epoch_data):
            self.index = 0
            self._shuffle_data()
        return l,t

    def setup(self, bottom, top):
        #print ("set up")
        file_names = glob.glob(P.FILENAMES_VALIDATION)
        self.file_true = filter(lambda x: "True" in x, file_names)
        self.file_false = filter(lambda x: "False" in x, file_names)
        self.n_true = len(self.file_true)
        self._shuffle_data()
        self.index = 0
        idx = 0
        top[idx].reshape(P.BATCH_SIZE_TRAIN, P.CHANNELS, PIXELS, PIXELS)
        idx += 1
        top[idx].reshape(P.BATCH_SIZE_TRAIN, W)

    def forward(self, bottom, top):
        #data, label = self.data_generator.next()
        data, label = self.read_data()
        if label.shape[0] != P.BATCH_SIZE_TRAIN:
            top[0].reshape(data.shape[0],P.CHANNELS, PIXELS, PIXELS)
            top[1].reshape(label.shape[0],W)
            print ('reshape ', label.shape[0])
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
        else:
            top[0].data[...] = data.astype(np.float32, copy=False)
            top[1].data[...] = label.astype(np.float32, copy=False)
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


