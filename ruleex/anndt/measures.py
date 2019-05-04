from abc import ABC

import numpy as np


def weighted_variance(y0, y1, w0, w1):
    """
    depracated
    Impurity - variance
    :param y0: probabilistic output for samples in a set 0
    :param y1: probabilistic output for samples in a set 1
    :param w0: weight of the set 0 impurity
    :param w1: weight of the set 1 impurity
    :return: weighted impurity - variance
    """
    return (w0 * np.var(y0[:, 0]) + w1 * np.var(y1[:, 0]))


def entropy(y0, y1, w0, w1):
    """
    depracated
    Impurity - entropy
    :param y0: probabilistic output for samples in a set 0
    :param y1: probabilistic output for samples in a set 1
    :param w0: weight of the set 0 impurity
    :param w1: weight of the set 1 impurity
    :return: weighted impurity - entropy
    """
    _, p0 = np.unique(np.argmax(y0, axis=1), return_counts=True)
    p0 = p0 / np.sum(p0)
    _, p1 = np.unique(np.argmax(y1, axis=1), return_counts=True)
    p1 = p1 / np.sum(p1)
    return - w0 * np.sum(p0 * np.log2(p0)) - w1 * np.sum(p1 * np.log2(p1))


def gini(y0, y1, train_len, w0, w1):
    """
    depracated
    Impurity - gini
    :param y0: probabilistic output for samples in a set 0
    :param y1: probabilistic output for samples in a set 1
    :param w0: weight of the set 0 impurity
    :param w1: weight of the set 1 impurity
    :return: weighted impurity - gini
    """
    _, p0 = np.unique(np.argmax(y0, axis=1), return_counts=True)
    p0 = p0 / np.sum(p0)
    _, p1 = np.unique(np.argmax(y1, axis=1), return_counts=True)
    p1 = p1 / np.sum(p1)
    return w0 * np.sum(p0 * (1 - p0)) + w1 * np.sum(p1 * (1 - p1))


def mean_max_class_difference(y0, y1, w0, w1):
    """
    depracated
    Impurity - maximal class difference
    :param y0: probabilistic output for samples in a set 0
    :param y1: probabilistic output for samples in a set 1
    :param w0: weight of the set 0 impurity
    :param w1: weight of the set 1 impurity
    :return: weighted impurity - maximal class difference
    """
    v0, c0 = np.unique(np.argmax(y0, axis=1), return_counts=True)
    c0 = v0[np.argmax(c0)]
    v1, c1 = np.unique(np.argmax(y1, axis=1), return_counts=True)
    c1 = v0[np.argmax(c1)]
    return w0 * np.mean(1-y0[:, c0]) + w1 * np.mean(1-y1[:, c1])


def fidelity_gain(y0, y1, w0, w1):
    l0 = np.argmax(y0, axis=1)
    v0, c0 = np.unique(l0, return_counts=True)
    c0 = v0[np.argmax(c0)]
    l1 = np.argmax(y1, axis=1)
    v1, c1 = np.unique(l1, return_counts=True)
    c1 = v0[np.argmax(c1)]
    return w0 * np.mean(l0!=c0) + w1 * np.mean(l1!=c1)


class MeasureOptimized(ABC):
    """
    Representation of the optimised process of finding the threshold by impurity
    """
    def get_mat(self, x, y, labels=None):
        """
        :param x:
        :param y:
        :param labels:
        :return: initial matrix which is main objective of the optimized process
        """
        if isinstance(labels, type(None)):
            labels = np.argmax(y, axis=1)
        u, c = np.unique(labels, return_counts=True)
        r0 = np.zeros(y.shape[1])
        r0[u] = c
        return np.array([r0, 0*r0])

    def get_weights(self, ts0, ts1):
        """
        :param ts0: the number of testing samples in a set 0
        :param ts1: the number of testing samples in a set 1
        :return: pair of an impurity weights
        """
        t = ts0/(ts0+ts1)
        return t, 1-t

    def measure(self, mat, counts, w0, w1):
        """
        The main function that needs to be implemented. It defines how the impurity is computed in each step, i.e.,
        how the value is transefer in the matrix
        :param mat: a matrix from which the impurity is computed
        :param counts: a number of samples
        :param w0: weight of the set 0
        :param w1: weight of the set 1
        :return: measure for the slit
        """
        pass

    def find_split(self, a, x, y, is_train_mask=None, apply_weights=True):
        """
        It searches for the best plit for the attribute a in the data x
        with probabilistic output y
        :param a: selected attribute index
        :param x: input values
        :param y: probabilistic output for x of the model
        :param is_train_mask: a mas of the samples which defines which are train (True) and which are generated (False)
        :param apply_weights: defines weather the weights of impurity should be used
        :return: the best split threshold for attribute
        """
        if not np.any(is_train_mask):
            is_train_mask = np.ones(x.shape[0], dtype=np.bool)
        xa = x[:,a]
        if np.all(xa[0] == xa):
            return np.inf, np.nan
        indeces = np.argsort(xa)
        xa = xa[indeces]
        labels = np.argmax(y, axis=1)[indeces]
        y = y[indeces]
        is_train_mask = is_train_mask[indeces]
        mat = self.get_mat(x, y, labels)
        th = xa[0]
        measures = list()
        thresholds = [th]
        counts = [len(x),0]
        for i, value in enumerate(xa):
            if th != value:
                th = value
                thresholds.append(th)
                if apply_weights:
                    w0, w1 = self.get_weights(np.sum(is_train_mask[i:]), np.sum(is_train_mask[:i]))
                else:
                    w0, w1 = 1, 1
                measures.append(self.measure(mat, counts, w0, w1))
            mat[0, labels[i]] -= 1
            mat[1, labels[i]] += 1
            counts[0] -= 1
            counts[1] += 1

        ind = np.argmin(measures)
        return measures[ind], (thresholds[ind+1]+thresholds[ind])/2


class GiniMeasure(MeasureOptimized):
    """
    Implementation of the Gini index impurity
    """
    def measure(self, mat, counts, w0, w1):
        p0 = mat[0, :] / counts[0]
        p1 = mat[1, :] / counts[1]
        return w0 * np.sum(p0 * (1 - p0)) + w1 * np.sum(p1 * (1 - p1))


class EntropyMeasure(MeasureOptimized):
    """
    Implementation of the Entropy impurity
    """
    def measure(self, mat, counts, w0, w1):
        p0 = mat[0, :] / counts[0]
        p0 = p0[p0>0]
        p1 = mat[1, :] / counts[1]
        p1 = p1[p1 > 0]
        return - w0 * np.sum(p0 * np.log2(p0)) - w1 * np.sum(p1 * np.log2(p1))


class FidelityGain(MeasureOptimized):
    """
    Implementation of the fidelity gain
    """
    def measure(self, mat, counts, w0, w1):
        return w0 * (1 - np.max(mat[0])/counts[0]) + w1 * (1 - np.max(mat[1])/counts[1])


class MaxDifference(MeasureOptimized):
    """
    Implementation of the maximal class difference
    """
    def get_mat(self, x, y, labels=None):
        r0 = np.sum(1 - y, axis=0)
        return np.array([r0, 0*r0])

    def measure(self, mat, counts, c0, c1, w0, w1):
        return w0 * (1 - mat[0, c0]/counts[0]) + w1 * (1 - mat[1,c1]/counts[1])

    def find_split(self, a, x, y, is_train_mask=None, apply_weights=True):
        if not np.any(is_train_mask):
            is_train_mask = np.ones(x.shape[0], dtype=np.bool)
        xa = x[:, a]
        if np.all(xa[0] == xa):
            return np.inf, np.nan
        indeces = np.argsort(xa)
        xa = xa[indeces]
        labels = np.argmax(y, axis=1)[indeces]
        y = y[indeces]
        is_train_mask = is_train_mask[indeces]
        mat = self.get_mat(x, y, labels)
        matc = np.zeros(mat.shape)
        th = xa[0]
        measures = list()
        thresholds = [th]
        counts = [len(x),0]
        for i, value in enumerate(xa):
            if th != value:
                th = value
                thresholds.append(th)
                if apply_weights:
                    w0, w1 = self.get_weights(np.sum(is_train_mask[i:]), np.sum(is_train_mask[:i]))
                else:
                    w0, w1 = 1, 1
                measures.append(self.measure(mat, counts, np.argmax(matc[0]), np.argmax(matc[1]), w0, w1))
            mat[0, :] -= y[i]-1
            mat[1, :] += y[i]-1
            matc[0, labels[i]] -= 1
            matc[1, labels[i]] += 1
            counts[0] -= 1
            counts[1] += 1

        ind = np.argmin(measures)
        return measures[ind], (thresholds[ind+1]-thresholds[ind])/2


class VarianceMeasure(MeasureOptimized):
    """
    Implementation of the variance impurity
    """
    def get_mat(self, x, y, labels=None):
        r1 = np.sum(y[:, 0])
        r2 = np.sum(y[:, 0]**2)
        return np.array([[r1, r2], [0., 0.]])

    def measure(self, mat, counts, w0, w1):
        return w0 * (mat[0,1] - mat[0,0]**2/counts[0])/counts[0] + w1 * (mat[1,1] - mat[1,0]**2/counts[1])/counts[1]

    def find_split(self, a, x, y, is_train_mask=None, apply_weights=True):
        if not np.any(is_train_mask):
            is_train_mask = np.ones(x.shape[0], dtype=np.bool)
        xa = x[:, a]
        if np.all(xa[0] == xa):
            return np.inf, np.nan
        indeces = np.argsort(xa)
        xa = xa[indeces]
        labels = np.argmax(y, axis=1)[indeces]
        y = y[indeces]
        is_train_mask = is_train_mask[indeces]
        mat = self.get_mat(x, y, labels)
        th = xa[0]
        measures = list()
        thresholds = [th]
        counts = [len(x),0]
        for i, value in enumerate(xa):
            if th != value:
                th = value
                thresholds.append(th)
                if apply_weights:
                    w0, w1 = self.get_weights(np.sum(is_train_mask[i:]), np.sum(is_train_mask[:i]))
                else:
                    w0, w1 = 1, 1
                measures.append(self.measure(mat, counts, w0, w1))
            mat[0, 0] -= y[i][0]
            mat[0, 1] -= y[i][0]**2
            mat[1, 0] += y[i][0]
            mat[1, 1] += y[i][0] ** 2
            counts[0] -= 1
            counts[1] += 1

        ind = np.argmin(measures)
        return measures[ind], (thresholds[ind+1]-thresholds[ind])/2
