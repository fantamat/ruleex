import numpy as np
import tensorflow as tf
import os

from .ruletree import RuleTree
from .rule import LinearRule
from .build import get_leaf
from gtrain.model import Model
from gtrain.data import AllData, BatchedData
from gtrain import gtrain
from gtrain.utils import confmat


class LinearModel(Model):
    def __init__(self, input_size, class_num,
                 impurity="entropy",
                 p=0.5):
        self.a = None
        self.b = None
        self.tf_a = None
        self.tf_b = None
        self.input_size = input_size
        self.class_num = class_num
        self.loss = None #
        self.x = None
        self.labels = None
        self.impurity_sum = None #
        self.count = None #
        self.impurity_mode = impurity
        self.p = p

    def build(self):
        """
        method that build whole model in a Tensorflow environment
        """
        if self.a is None or self.b is None:
            raise Exception("Attributes a and b of a object of the LinearModel class must be initialized "
                            "before running build method or gtrain function!")
        self.tf_a = tf.Variable(self.a, name="a")
        self.tf_b = tf.reshape(tf.Variable(self.b, dtype=tf.float64, name="b"), (1,))
        self.x = tf.placeholder(tf.float64, shape=(None, self.input_size), name="input")
        self.labels = tf.placeholder(tf.int64, shape=(None, 1), name="labels")

        #
        self.count = tf.cast(tf.size(self.labels), tf.float64)
        s_t = tf.sigmoid(tf.nn.xw_plus_b(self.x, self.tf_a, self.tf_b))
        s_f = 1. - s_t
        sum_t = tf.reduce_sum(s_t) + self.class_num
        sum_f = tf.reduce_sum(s_f) + self.class_num
        zeros = tf.zeros_like(s_t)
        sum_t_j = tf.stack([tf.reduce_sum(tf.where(self.labels == j, s_t, zeros)) + 1 for j in range(self.class_num)])
        sum_f_j = tf.stack([tf.reduce_sum(tf.where(self.labels == j, s_f, zeros)) + 1 for j in range(self.class_num)])

        # absolute values of a sums to one
        absone = (1 - tf.reduce_sum(tf.abs(self.tf_a)))**2 / 2 / self.input_size
        absone_coeff = (1 - self.p) / (1 + np.log2(self.class_num))

        # values of a are either close to 1, -1, or 0
        zeroone = tf.reduce_mean(tf.abs(self.tf_a*(1 - self.tf_a)))
        zeroone_coeff = absone_coeff

        # number of samples on both sides of hyperplane are balanced
        sides_balanced = (sum_t - sum_f)**2 / 2 / self.count
        sides_balanced_coeff = absone_coeff * (np.log2(self.class_num) - 1) / 100000

        # impurity
        if self.impurity_mode == "gini":
            impurity = (sum_t - tf.reduce_sum(sum_t_j)**2) / sum_t + \
                       (sum_f - tf.reduce_sum(sum_f_j) ** 2) / sum_f
        else:
            impurity = (- tf.reduce_sum(sum_t_j * tf.log(sum_t_j / sum_t))) + \
                       (- tf.reduce_sum(sum_f_j * tf.log(sum_f_j / sum_f)))

                # old too big
                #sum_t * (sum_t * tf.log(sum_t) - tf.reduce_sum(sum_t_j * tf.log(sum_t_j))) + \
                #       sum_f * (sum_f * tf.log(sum_f) - tf.reduce_sum(sum_f_j * tf.log(sum_f_j)))
        self.impurity_sum = impurity
        impurity = impurity / self.count
        impurity_coeff = self.p / np.log(self.class_num)

        #loss
        self.loss = (#absone_coeff * absone + # remake to have maximal coeff with abs value 1
                    #zeroone_coeff * zeroone +
                    #sides_balanced_coeff * sides_balanced +
                    impurity_coeff * impurity)



    def get_loss(self):
        """
        :return: a tf variable that represents the loss function
        """
        return self.loss

    def get_hits(self):
        """
        :return: a tf variable that is the impurity measure multiplied by the number of samples
        """
        return self.impurity_sum

    def get_count(self):
        """
        :return: a tf variable that represents the number of samples
        """
        return self.count

    def get_train_summaries(self):
        """
        :return: additional summaries that are computed on the training data at each step of the gtrain training cycle
        """
        return []

    def get_dev_summaries(self):
        """
        :return: additional summaries that are computed on the testing data at each step of the gtrain training cycle
        """
        return []

    def get_placeholders(self):
        """
        :return: a tf placeholders of the input and output of the model
        """
        return [self.x, self.labels]

    def train_ended(self, session):
        """
        a method that is called after the training of the model ended
        - you can for example save the trained weights of the model
        """
        self.a, self.b = session.run([self.tf_a, self.tf_b])

    def name(self):
        """
        :return: a name of the model with the particular settings
        """
        return "Linear model for oblique decision tree"


class ODTLinearOptimizer:
    def __init__(self, class_num, input_size, out_dir=None, **kwargs):
        self.n = input_size
        self.c = class_num
        self.a = np.zeros(input_size)
        self.b = 0
        self.node_counter = 0
        self.depth = 0
        self.find_optimal_counter = 0
        self.out_dir = out_dir
        self.model = LinearModel(input_size, class_num, **kwargs)

    def next_node(self, depth):
        self.node_counter += 1
        self.depth = depth

    def get_current_out_dir(self):
        return os.path.join(self.out_dir,
                            "depth_{:02d}".format(self.depth),
                            "node_{:02d}-step_{:02d}".format(self.node_counter, self.find_optimal_counter))

    def find_optimal(self, x, labels, a, b, **kwargs):
        self.find_optimal_counter += 1
        self.model.a = a
        self.model.b = b
        labels = labels.reshape((len(labels),1))
        data = AllData(x, labels, x, labels)
        if self.out_dir is not None:
            kwargs["out_dir"] = self.get_current_out_dir()
        gtrain(self.model, data, dtype=tf.float64, **kwargs)
        print("a,b diff:")
        print((self.model.a-a).flatten(), self.model.b-b)
        return self.model.a, self.model.b


def build_odt(x, labels,
              threshold_mode="value",
              threshold_value=0.01,
              max_depth=5,
              min_samples=1,
              min_split_fraction: float = 0.02,
              impurity="entropy",
              p=0.5,
              out_dir=None):
    """
    Builds oblique DT by linear optimization
    :param x: inputs of training samples
    :param labels: labels of training samples
    :param threshold_mode: three types of threshold:
        "value": uses threshold_value as static constant
        "hard": uses dynamic threshold as max(a)*threshold_value
        "soft": uses dynamic threshold as 1/z*threshold_value
    :param threshold_value: threshold of the coefficient that is considered as zero
    :return: oblique RuleTree
    """
    gtrain_params = {
        "num_steps": 10000,
        "evaluate_every": 1000,
        "checkpoint_every": 1000,
    }
    num_class = max(labels) + 1
    input_size = x.shape[1]
    output = RuleTree(num_class, input_size)
    optimizer = ODTLinearOptimizer(num_class, input_size, out_dir=out_dir, impurity=impurity, p=p)

    if threshold_mode=="soft":
        get_threshold = lambda a, z: 1 / z * threshold_value
    elif threshold_mode=="hard":
        get_threshold = lambda a, z: max(a) * threshold_value
    else:
        get_threshold = lambda a, z: threshold_value

    def __build_odt_node(x, labels, depth):
        if depth == max_depth:
            print("[ODT]: Maximal depth reached (max_depth={}).".format(max_depth))
            return get_leaf(labels, num_class)

        if len(labels) <= min_samples:
            print("[ODT]: Node contains less than min_samples={}.".format(min_samples))
            return get_leaf(labels, num_class)

        a, b, = np.random.randn(input_size, 1), 0
        z, z_old = input_size, input_size + 1
        optimizer.next_node(depth)
        while z != z_old:
            a, b = optimizer.find_optimal(x, labels, a, b, **gtrain_params)
            threshold = get_threshold(a, z)
            a[np.abs(a) <= threshold] = 0
            z_old = z
            z = np.sum(a==0)
            node = LinearRule(a.flatten(), - b[0])
            filter = node.eval_all(x)
            _, c0 = np.unique(labels[filter], return_counts=True)
            _, c1 = np.unique(labels[~filter], return_counts=True)
            print("data division {}".format(np.mean(filter)))
            print("branches occupation True: {} False {}".format(c0, c1))
        if np.all(filter) or not np.any(filter):
            print("[ODT]: Split do not divide training data.")
            return get_leaf(labels, num_class)
        if min(np.mean(filter), 1 - np.mean(filter)) < min_split_fraction:
            print("[ODT]: Split divide training data in less than min_plit_fraction={}!".format(min_split_fraction))
            return get_leaf(labels, num_class)
        node.true_branch = __build_odt_node(x[filter], labels[filter], depth + 1)
        node.false_branch = __build_odt_node(x[~filter], labels[~filter], depth + 1)
        return node

    output.root = __build_odt_node(x, labels, 0)
    return output
