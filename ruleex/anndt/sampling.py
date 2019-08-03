from abc import ABC

import numpy as np


class Sampler(ABC):
    """
    Abstraction of the sampler for ANN-DT algorithm
    """
    def get_default_params(self, x):
        """
        :return: default params for given sampler
        """
        pass

    def generate_x(self, train_x, number, restrictions):
        """
        Generates new samples
        :param train_x: available train_samples for given node
        :param number: the number of samples that should be generated
        :param restrictions: restriction to the generated samples (limit for each attribute)
        :return: randomly generated samples
        """
        pass

    def apply_restrictions(self, sample, restrictions):
        """
        :param sample: a sample
        :param restrictions: restriction to the generated samples (limit for each attribute)
        :return: True if the sample fits the restrictions
        """
        return np.all(sample >= restrictions[:, 0]) and np.all(sample <= restrictions[:, 1])


class NormalSampler(Sampler):
    """
    Representation of the normal sampler.
    It uses normal kernel distribution to generate new samples form the training samples
    """
    def __init__(self, x, additional_samples=0.5):
        """
        :param x: training samples from which a covariance matrix is computed
        :param additional_samples: a fraction that is additionaly generated at each step
            - it helps if the restrictions are less often fulfilled because more samples are generated and tested
        """
        self.cov = np.cov(np.swapaxes(x, 0, 1))
        self.additional_samples = additional_samples
        self.attr_num = len(x[0])

    def generate_x(self, train_x, count, restrictions):
        try_count = round(count*(1 + self.additional_samples))
        out = np.empty((0, self.attr_num))
        i = 0
        while len(out) < count:
            ri = np.random.randint(0, len(train_x), try_count)
            gens = np.random.multivariate_normal(np.zeros(self.attr_num), self.cov, try_count)
            gens = gens + train_x[ri]
            gens = gens[np.apply_along_axis(self.apply_restrictions, 1, gens, restrictions)]
            out = np.append(out, gens, 0)
            i += 1
            if i > 20:
                print("[anndt]: Generated only {} samples maximal count of the generationg cycle exceeds the threshold 20".format(len(out)))
                break
        return out[:count]


class BerNormalSampler(Sampler):
    """
    Representation of the bernuli*normal distribution with parameters p, mu, sigma
    """
    def __init__(self, x, additional_samples=0.5, always_positive=False, sigma=None):
        """
        :param x: training samples from which is taken p as a mean(x==0) and sigma = std(x) if not defined
        :param additional_samples: a fraction that is additionaly generated at each step
            - it helps if the restrictions are less often fulfilled because more samples are generated and tested
        :param always_positive: a flag that defines weather samples should be always positive
        :param sigma: a parameter of the kernel distribution
        """
        self.p = np.mean(x==0, 0)
        if sigma is None:
            self.sigma = np.std(x, 0)
        else:
            self.sigma = sigma
        self.additional_samples = additional_samples
        self.attr_num = len(x[0])
        self.always_positive = always_positive

    def generate_x(self, train_x, count, restrictions):
        try_count = round(count * (1 + self.additional_samples))
        out = np.empty((0, self.attr_num))
        i = 0
        while len(out) < count:
            ri = np.random.randint(0, len(train_x), try_count)
            zeros = np.apply_along_axis(np.less, 1, np.random.rand(try_count, self.attr_num), self.p)
            gens = self.sigma * np.random.randn(try_count, self.attr_num) * zeros
            gens = gens + train_x[ri]
            gens[train_x[ri]==0] = 0
            if self.always_positive:
                gens = np.abs(gens)
            gens = gens[np.apply_along_axis(self.apply_restrictions, 1, gens, restrictions)]
            out = np.append(out, gens, 0)
            i += 1
            if i > 20:
                print("[anndt]: Generated only {} samples maximal count of the generationg cycle exceeds the threshold 20".format(len(out)))
                break
        return out[:count]
