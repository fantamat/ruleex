
import numpy as np
import ruleex.tree.utils as rtu
from ruleex.hypinv import hypinv, INF
from ruleex.hypinv.model import FCNetForHypinvBinary, FCNetForHypinv

MODEL_PARAMS = "model_params"

def hypinv_class_wise(weights, x, params, inputRange=None):
    """
    Uses weights to initialize FCnetForHypinvBynary for each class
    """
    if MODEL_PARAMS not in params:
        params[MODEL_PARAMS] = dict()

    def hypinv_one_class(class_index, train_indexes):
        model = FCNetForHypinvBinary(weights, base_class_index=class_index, **params[MODEL_PARAMS])
        rt = hypinv(model, x, params, inputRange)
        del model
        return rt, params[INF]

    model = FCNetForHypinv(weights)
    l = np.argmax(model.eval(x), axis=1)
    del model
    out_rt, out_inf = rtu.train_class_wise(l, hypinv_one_class)
    return out_rt, out_inf

