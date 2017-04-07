from __future__ import division
import numpy as np

def weight_by_class_balance(truth, classes = None):
    """
    Determines a loss weight map given the truth by balancing the classes from the classes argument.
    The classes argument can be used to only include certain classes (you may for instance want to exclude the background).
    """

    if classes is None:
        # Include all classes
        classes = np.unique(truth)

    weight_map = np.zeros_like(truth, dtype = np.float32)
    # total_amount = np.product(truth.shape)

    count = []
    for c in classes:
      count.append(np.sum(truth == c))
    total_amount = np.sum(np.array(count))

    for c in classes:
        class_mask = np.where(truth == c, 1, 0)
        class_weight = 1 / ((np.sum(class_mask) + 1e-8) / total_amount)
        # print "=========== class: ", c
        # print "class weight: ", class_weight
        # print "class num: ", np.sum(class_mask)
        # print "total amount: ", total_amount
        # assert (np.sum(class_mask) > 0)

        weight_map += (class_mask * class_weight)

    return weight_map

if __name__ == "__main__":
    x = np.array([[1, 0, 0], [-1, 1, 0]])
    print weight_by_class_balance(x, [0, 1])
