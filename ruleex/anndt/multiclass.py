
import numpy as np
import ruleex.tree.utils as rtu
from ruleex.anndt import anndt
from ruleex.deepred import deepred, INF


def anndt_class_wise(model, x, params, MeasureClass=None, stat_test=None, sampler=None, init_restrictions=None):
    """

    :param model: model that implements method eval_binary_class(self, x, class_index), e.g., deepred.model.DeepRedFCNet
    :param x: training inputs
    :param params: anndt parameters
    :param MeasureClass: anndt's measure class
    :param stat_test: statistical test on nodes
    :param sampler: sampler for new samples
    :param init_restrictions: initial restrictions
    :return:
    """
    # def anndt(model_fun, x, params, MeasureClass=None, stat_test=None, sampler=None, init_restrictions=None):
    def anndt_one_class(class_index, train_indexes):
        rt = anndt(lambda x: model.eval_binary_class(x, class_index),
                   x[train_indexes],
                   params,
                   MeasureClass,
                   stat_test,
                   sampler,
                   init_restrictions)
        return rt, params[INF]
    l = np.argmax(model.eval(x), axis=1)
    out_rt, out_inf = rtu.train_class_wise(l, anndt_one_class)
    return out_rt, out_inf

