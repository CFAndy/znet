import _init_paths

import caffe
import numpy as np
import sys
import glob

class EvalLayer(caffe.Layer):

    def setup(self, bottom, top):
        # print ("setup")
        top[0].reshape(4)

    def forward(self, bottom, top):
        count = np.product(bottom[1].data[...].shape)
        prob = bottom[0].data[...].flatten()
        label = bottom[1].data[...].flatten()
        prediction = np.zeros_like(prob, dtype = np.float32)
        prediction[prob > 0.5] = 1
        
        tp = np.sum(np.equal(label, 1) * np.equal(prediction, 1))
        tn = np.sum(np.not_equal(label, 1) * np.not_equal(prediction, 1))
        fp = np.sum(np.not_equal(label, 1) * np.equal(prediction, 1))
        fn = np.sum(np.equal(label, 1) * np.not_equal(prediction, 1))

        accuracy = (tp + tn) / count
        specificity = tn / (fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        top[0].data[0] = accuracy
        top[0].data[1] = specificity
        top[0].data[2] = precision
        top[0].data[3] = recall

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
