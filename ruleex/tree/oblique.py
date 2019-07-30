import numpy as np

from .ruletree import RuleTree
from .rule import LinearRule, Leaf


def find_optimal(x, labels, a, b, session=None):
    #todo
    return a, b


def build_odt(x, labels,
              threshold_mode="value",
              threshold_value=0.01):
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
    num_class = max(labels) + 1
    input_size = x.shape[1]
    output = RuleTree(num_class, input_size)

    if threshold_mode=="soft":
        get_threshold = lambda a, z: 1 / z * threshold_value
    elif threshold_mode=="hard":
        get_threshold = lambda a, z: max(a) * threshold_value
    else:
        get_threshold = lambda a, z: threshold_value

    def __build_odt_node(x, labels):
        a, b, = np.random.randn(input_size), 0
        z, z_old = input_size, input_size + 1
        while z != z_old:
            a, b = find_optimal(x, labels, a, b)
            threshold = get_threshold(a, z)
            a = [ai if abs(ai) > threshold else 0 for ai in a]
            z_old = z
            z = np.sum(a==0)
        node = LinearRule(a, b)
        filter = node.eval_all(x)
        #todo stopping rule
        node.true_branch = __build_odt_node(x[filter], labels[filter])
        node.false_branch = __build_odt_node(x[~filter], labels[~filter])
        return node


    output.root = __build_odt_node(x, labels)
    return output