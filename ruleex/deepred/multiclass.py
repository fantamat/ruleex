
import numpy as np
import ruleex.tree.utils as rtu
from ruleex.deepred import deepred, INF


def deepred_class_wise(layers_activations, params, append_last=False):
    """
    generate ddag from the multiclass problem class-wise by using function train_class_wise from ruleex.tree.utils
    :param layers_activations: activation of the neural network final activations are positive and are summed to one
    :param params: params passed into the deepred function
    :param append_last: if True then the created layer for one vs the others classification is appended
        after the original last one. Otherwise the last layer is substituted
    :return: ruletree, inf_dict
    """
    def deepred_one_class(class_index, train_indexes):
        activations = list()
        for activation in layers_activations:
            activations.append(activation[train_indexes])
        last = np.zeros((len(train_indexes), 2))
        last[:, 0] = activations[-1][:, class_index]
        mask = np.ones(len(activations[-1][0]), dtype=np.bool)
        mask[class_index] = False
        last[:, 1] = np.max(activations[-1][:, mask], axis=1)
        s = last[:, 0] + last[:, 1]
        last[:, 0] = last[:, 0] / s
        last[:, 1] = last[:, 1] / s
        if append_last:
            activations.append(last)
        else:
            activations[-1] = last
        rt = deepred(activations, params)
        return rt, params[INF]
    l = np.argmax(layers_activations[-1], axis=1)
    out_rt, out_inf = rtu.train_class_wise(l, deepred_one_class)
    return out_rt, out_inf

