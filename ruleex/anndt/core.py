import time
from threading import Thread

from ruleex.anndt.measures import *
from ruleex.anndt.sampling import *
from ruleex.anndt.stat_test import *
from ruleex.tree.rule import AxisRule, Leaf
from ruleex.tree.ruletree import RuleTree
import numpy as np

from ruleex.utils import fill_default_params


INF = "inf"
TIME = "time"
GENERATED_SAMPLES = "generated_samples" # dictionary node->number of generated samples
STOPPED_BY_RULE = "stopped_by_rule" # list of nodes in which process were stopped by the stopping criteria
STOPPED_BY_TEST = "stopped_by_test" # list of nodes in which process were stoppped by applied statistical test
USED_INPUTS = "used_inputs" # if the attribute selection is applied that it specified the indexes that were selected to use

VARBOSE = "varbose"
DEFAULT_VARBOSE = 1
MIN_SAMPLES = "min_samples"
DEFAULT_MIN_SAMPLES = 50
FORCE_SAMPLING = "force_sampling"
DEFAULT_FORCE_SAMPLING = False
MIN_TRAIN_SAMPLES = "min_train"
DEFAULT_MIN_TRAIN_SAMPLES = 2
MAX_DEPTH = "max_depth"
DEFAULT_MAX_DEPTH = 6
SPLIT_TEST_AFTER = "test_after"
DEFAULT_SPLIT_TEST_AFTER = 3
SPLIT_TEST_LEVEL = "test_level"
DEFAULT_SPLIT_TEST_LEVEL = 0.05
INTEGRAL_SEPARATION_NUM = "separation_num"
DEFAULT_INTEGRAL_SEPARATION_NUM = 10
STOP_MEAN_VARIANCE = "stop_mean_variance"
DEFAULT_STOP_MEAN_VARIANCE = 0
ATTRIBUTE_SELECTION = "attribute_selection"
MODE_ABSOLUTE_VARIATION = "absolute_variation"
MODE_MISSCLASSIFICATION = "missclasificaiton"
MODE_CONTINUOUS_MISSCLASSIFICATION = "continuous_missclasificaiton"
MODE_AS_THRESHOLD = "as_threshold"
DEFAULT_ATTRIBUTE_SELECITON = MODE_AS_THRESHOLD
MIN_SPLIT_FRACTION = "min_split_fraction"
DEFAULT_MIN_SPLIT_FRACTION = 0.02

INPUT_PRUNING = "input_pruning"
MISSCLASSIFICATION_PRUNING = "misclassification"
CONTINUOUS_MISSCLASSIFICATION_PRUNING = "continuous_misclassification"
NO_PRUNING = "none"
DEFAULT_PRUNING = NO_PRUNING
MEASURE_WEIGHTS = "measure_weights"
MODE_TRAIN = "mode_train"
MODE_ALL = "mode_all"
MODE_NONE = "mode_none"
DEFAULT_MEASURE_WEIGHTS = MODE_TRAIN

DEFAULT_PARAMS = {
    VARBOSE: DEFAULT_VARBOSE,
    MIN_SAMPLES: DEFAULT_MIN_SAMPLES,
    MIN_TRAIN_SAMPLES: DEFAULT_MIN_TRAIN_SAMPLES,
    MAX_DEPTH: DEFAULT_MAX_DEPTH,
    SPLIT_TEST_AFTER: DEFAULT_SPLIT_TEST_AFTER,
    INTEGRAL_SEPARATION_NUM: DEFAULT_INTEGRAL_SEPARATION_NUM,
    STOP_MEAN_VARIANCE: DEFAULT_STOP_MEAN_VARIANCE,
    ATTRIBUTE_SELECTION: DEFAULT_ATTRIBUTE_SELECITON,
    SPLIT_TEST_LEVEL: DEFAULT_SPLIT_TEST_LEVEL,
    MEASURE_WEIGHTS: DEFAULT_MEASURE_WEIGHTS,
    INPUT_PRUNING: DEFAULT_PRUNING,
    MIN_SPLIT_FRACTION: DEFAULT_MIN_SPLIT_FRACTION,
    FORCE_SAMPLING: DEFAULT_FORCE_SAMPLING,
}


def missclass_pruning(x, model_fun, continuous=False, return_missclassificaiton=False):
    """
    Return sorted indexes with respect to their missclassificaiton rate
    :param x: input samples
    :param model_fun: model function
    :param continuous: switcher between continuous and discrete missclassificaiton
    :param return_missclassificaiton: if True then also vector of missclassificaiton values is returned
    :return: sorted indexes with respect to their missclassificaiton rate
    """
    y = model_fun(x)
    class_num = len(y[0])
    attr_num = len(x[0])
    u, c = np.unique(np.argmax(y, axis=1), return_counts=True)
    p = np.zeros(class_num)
    p[u] = c / np.sum(c)
    # p = np.sum(np.append(train_y, gen_y, 0), axis=0)
    # p = p/np.sum(p)
    missclassificaiton = np.zeros(attr_num)
    for a in range(attr_num):
        ax = x.copy()
        ax[:, a] = 0
        ay = model_fun(ax)
        if continuous:
            missclassificaiton[a] = np.dot(np.sum((y - ay) ** 2, 0), p)
        else:
            missclassificaiton[a] = np.mean(np.argmax(y, 1) != np.argmax(ay, 1))
    if return_missclassificaiton:
        return np.argsort(missclassificaiton), missclassificaiton
    else:
        return np.argsort(missclassificaiton)


def anndt(model_fun, x, params, MeasureClass=None, stat_test=None, sampler=None, init_restrictions=None):
    """
    ANN-DT algorith of the rule extraction from the model_fun
    :param model_fun: model with probabilistic output
    :param x: train samples
    :param params: a dictionary with additional parameters (keys are defined as constant of this module)
        VARBOSE: level of the logs into the stdout
        MIN_SAMPLES: minimal number of samples both training and generated in the node
            if the number is lower then new samples are generated
        MIN_TRAIN_SAMPLES: minimal number of training samples in the node
        MAX_DEPTH: maximal depth of the generated RuleTree
        SPLIT_TEST_AFTER: from what depth is started to test the split set by statistical test
        INTEGRAL_SEPARATION_NUM: number of separations used in the process of computing the absolute variation
        STOP_MEAN_VARIANCE: minimal mean variance if the samples variance si lower the process stops (stopping rule)
        ATTRIBUTE_SELECTION: defines attribute selection method (None or one of MODE_ABSOLUTE_VARIATION,
            MODE_MISSCLASSIFICATION)
        SPLIT_TEST_LEVEL: level of the statistical test
        MEASURE_WEIGHTS: defines which approach is used to weight measure to compute final impurity
            MODE_TRAIN: only train samples are considered to weight the measures
            MODE_ALL: both types of samples train and generated are considered to weight the measures
            MODE_NONE: weight are constant (set to 1)
        INPUT_PRUNING: defines method of the input prunning only if is defined and set to one of
            MISSCLASSIFICATION_PRUNING
            CONTINUOUS_MISSCLASSIFICATION_PRUNING
        MIN_SPLIT_FRACTION: mimimal ration of the splited samples by a new node
        FORCE_SAMPLING: if True then at each new node there are new samples generated
            the number of newly generated samples is defined by MIN_SAMPLES
    :param MeasureClass: subclass of MeasureOptimized
    :param stat_test: function with three arguments label_0, label_1, level see stat_test.py
    :param sampler: a instance of the subclass of MeasureOptimized see measures.py
    :return: extracted RuleTree object
        In addition, information about procees are stored as dictionary in the params[INF]
    """

    # fill required variables
    fill_default_params(DEFAULT_PARAMS, params)
    if not sampler:
        sampler = NormalSampler(x)
    if not MeasureClass:
        MeasureClass = EntropyMeasure
    if not stat_test:
        stat_test = test_F
    if init_restrictions is None:
        init_restrictions = np.array([[-np.inf, np.inf] for i in x[0]])
    init_restrictions = init_restrictions.astype(np.float)
    params[INF] = dict()
    y = model_fun(x)
    attr_num = x.shape[1]
    class_num = y.shape[1]
    inf = dict()
    inf[GENERATED_SAMPLES] = dict()
    inf[STOPPED_BY_RULE] = list()
    inf[STOPPED_BY_TEST] = list()
    tic = time.time()

    # definition of inner functions
    def get_absolute_variation(x0, x1):
        """
        computes absolute variation of the two samples x0 and x1
        """
        dx = (x1-x0)/params[INTEGRAL_SEPARATION_NUM]
        x = np.mgrid[:params[INTEGRAL_SEPARATION_NUM]+1, :attr_num][0]
        x = x0 + np.apply_along_axis(np.multiply, 1, x, dx)
        o = model_fun(x)[:, 0]
        o = np.diff(o)
        o = np.abs(o)
        return np.sum(o)

    def missclass_pruning(x, y, continuous=False):
        """
        Pruns the attributes based on the missclassification, more precisely only attributes which  provides
        the most difference in output if are set to zero are selected
        """
        u, c = np.unique(np.argmax(y, axis=1), return_counts=True)
        p = np.zeros(class_num)
        p[u] = c / np.sum(c)
        missclassificaiton = np.zeros(attr_num)
        for a in range(attr_num):
            ax = x.copy()
            ax[:, a] = 0
            ay = model_fun(ax)
            if continuous:
                missclassificaiton[a] = np.dot(np.sum((y - ay) ** 2, 0), p)
            else:
                missclassificaiton[a] = np.mean(np.argmax(y, 1) != np.argmax(ay, 1))
        if continuous:
            # takes best 25%
            indexes = np.argsort(missclassificaiton)
            x[:, indexes[:-len(indexes)//4]] = 0
            inf[USED_INPUTS] = indexes[-len(indexes)//4:]
        else:
            # takes all attributes that have more than one wrongly classified sample
            x[:, missclassificaiton <= 1/len(x)] = 0
            inf[USED_INPUTS] = np.arange(attr_num)[missclassificaiton > 1/len(x)]
        return x, model_fun(x)

    def find_by_absolut_variation(train_x, train_y, gen_x, gen_y):
        """
        Attribute selection method based on the absolut variation
        (correlation between absolut variation and feature difference)
        """
        u,c = np.unique(np.argmax(np.append(train_y, gen_y, 0), axis=1), return_counts=True)
        p = 1+np.zeros(class_num)
        p[u] = c/np.sum(c)
        significance = np.zeros(attr_num)
        x = np.append(train_x, gen_x, 0)
        if len(x) > 2**7: # restricts size of absolute_variation to the size of 4gb
            np.random.shuffle(x)
            x = x[:2**7]
        lx = len(x)
        absolute_variation = np.zeros(lx*(lx-1))
        for i in range(1,lx):
            for j in range(i):
                absolute_variation[j+(i-1)*i//2] = get_absolute_variation(x[i], x[j]) * p[0]
        for a in range(attr_num):
            attribute_variation = np.zeros(lx*(lx-1))
            for i in range(1, lx):
                for j in range(i):
                    attribute_variation[j + (i - 1) * i // 2] = x[j, a] - x[i, a]
            significance[a] = np.correlate(absolute_variation, attribute_variation)
        return np.argmax(significance)

    def find_by_missclassificaiton(train_x, train_y, gen_x, gen_y):
        """
        Attribute selection method base on missclassificaiton
        The attributes which generates the largest difference in the output by setting them to zero are selected
        """
        u,c = np.unique(np.argmax(np.append(train_y, gen_y, 0), axis=1), return_counts=True)
        p = np.zeros(class_num)
        p[u] = c/np.sum(c)

        x = np.append(train_x, gen_x, 0)
        y = model_fun(x)
        missclassificaiton = np.zeros(attr_num)
        for a in range(attr_num):
            ax = x.copy()
            ax[:, a] = 0
            ay = model_fun(ax)
            missclassificaiton[a] = np.dot(np.sum((y-ay)**2, 0), p)
        return np.argmax(missclassificaiton)

    def find_threshold(train_x, train_y, gen_x, gen_y, a=None):
        """
        Technical details of the finding the split threshold and if the attribute a is not defined then also
        an attribute of split
        """
        train_len = len(train_x)
        x = np.append(train_x, gen_x, 0)
        y = np.append(train_y, gen_y, 0)
        def process_attribute(a):
            mc = MeasureClass()
            if params[MEASURE_WEIGHTS] == MODE_TRAIN:
                is_train_mask = np.zeros(len(x), dtype=np.bool)
                is_train_mask[:train_len] = True
                return mc.find_split(a, x, y, apply_weights=True, is_train_mask=is_train_mask)
            elif params[MEASURE_WEIGHTS] == MODE_ALL:
                return mc.find_split(a, x, y, apply_weights=True)
            else:
                return mc.find_split(a, x, y, apply_weights=False)

        if a is not None:
            return process_attribute(a)[1]
        else:
            attrs_measurement = np.zeros(attr_num)
            attrs_best_split = np.zeros(attr_num)
            # parallel
            class thread_it(Thread):
                def __init__(self, param):
                    Thread.__init__(self)
                    self.param = param
                def run(self):
                    m, bs = process_attribute(self.param)
                    attrs_best_split[self.param] = bs
                    attrs_measurement[self.param] = m
            tl = list()
            for a in range(attr_num):
                current = thread_it(a)
                tl.append(current)
                current.start()
            for t in tl:
                t.join()
            """
            # not parallel
            for a in range(attr_num):
                attrs_measurement[a], attrs_best_split[a] = process_attribute(a)
                
                        """
            a = np.argmin(attrs_measurement)
            return a, attrs_best_split[a]

    def get_leaf(train_y, gen_y):
        """
        Creates the Leaf based on the model's outputs
        """
        labels = np.argmax(np.append(train_y, gen_y, 0), axis=1)
        values, counts = np.unique(labels, return_counts=True)
        c = np.zeros(class_num)
        c[values] = counts
        max_label = values[np.argmax(counts)]
        node = Leaf({max_label})
        node.description = "Train: {}".format(len(train_y))
        node.class_hits = c
        return node

    def stopping_rule(train_x, train_y, gen_x, gen_y, level):
        """
        stopping rules of the building algorithm
        """
        if level == params[MAX_DEPTH]:
            print("[anndt]: stopping rule - max depth exceeded ({})".format(level))
            return True
        if len(train_x)<params[MIN_TRAIN_SAMPLES]:
            print("[anndt]: stopping rule - low number of train samples")
            return True
        if np.mean(np.var(np.append(train_y, gen_y, 0))) <= params[STOP_MEAN_VARIANCE]:
            print("[anndt]: stopping rule - stopping mean variance ")
            return True
        return False

    def build_node(train_x, train_y, gen_x, gen_y, level, restrictions):
        """
        recursive building function
        """
        if stopping_rule(train_x, train_y, gen_x, gen_y, level):
            node = get_leaf(train_y, gen_y)
            inf[STOPPED_BY_RULE].append(node)
            return node
        gen_samples = None
        if len(train_x)+len(gen_x) < params[MIN_SAMPLES] or params[FORCE_SAMPLING]:
            if params[FORCE_SAMPLING]:
                gen_samples = params[MIN_SAMPLES]
            else:
                gen_samples = params[MIN_SAMPLES] - len(train_x) - len(gen_x)
            if params[VARBOSE]>0:
                print("[anndt]: Generating {} new samples.".format(gen_samples))
            new_gen_x = sampler.generate_x(np.append(train_x, gen_x, 0), gen_samples, restrictions)
            new_gen_y = model_fun(new_gen_x)
            gen_x = np.append(gen_x, new_gen_x, 0)
            gen_y = np.append(gen_y, new_gen_y, 0)
        if find_attribute:
            a = find_attribute(train_x, train_y, gen_x, gen_y)
            th = find_threshold(train_x, train_y, gen_x, gen_y, a)
        else:
            a, th = find_threshold(train_x, train_y, gen_x, gen_y)
        if np.isnan(a) or np.isnan(th):
            print("[anndt]: Selected attribute is NaN, i.e.,  do not split data.")
            return get_leaf(train_y, gen_y)
        node = AxisRule(a, th)
        if gen_samples:
            inf[GENERATED_SAMPLES][node] = gen_samples
        eval_train = node.eval_all(train_x)
        eval_gen = node.eval_all(gen_x)
        fraction = (np.sum(eval_train) + np.sum(eval_gen)) / (len(eval_train) + len(eval_gen))
        if min(fraction, 1-fraction) < params[MIN_SPLIT_FRACTION]:
            print("[anndt]: Stopping rule - fraction of the founded node is to low so the leaf is generated.")
            return get_leaf(train_y, gen_y)
        if level > params[SPLIT_TEST_AFTER] and \
                stat_test(np.append(train_y[eval_train], gen_y[eval_gen], 0),
                       np.append(train_y[~eval_train], gen_y[~eval_gen], 0),
                       params[SPLIT_TEST_LEVEL]):
            print("[anndt]: Statistics test passed at confidence level {}"
                  .format(params[SPLIT_TEST_LEVEL]))
            node = get_leaf(train_y, gen_y)
            inf[STOPPED_BY_TEST].append(node)
            return node
        if params[VARBOSE] > 0:
            sum_et = np.sum(eval_train)
            print("[anndt]: Generated new node with split x_{} > {} in train samples separation ({}, {})".format(
                a, th, sum_et, len(eval_train)-sum_et
            ))
        trestrictions = restrictions.copy()
        trestrictions[a, 0] = th
        node.true_branch = build_node(train_x[eval_train], train_y[eval_train],
                                      gen_x[eval_gen], gen_y[eval_gen],
                                      level+1, trestrictions)

        frestrictions = restrictions.copy()
        frestrictions[a, 1] = th
        node.false_branch = build_node(train_x[~eval_train], train_y[~eval_train],
                                       gen_x[~eval_gen], gen_y[~eval_gen],
                                       level+1, frestrictions)
        return node

    # fill defaults
    if params[INPUT_PRUNING] == MISSCLASSIFICATION_PRUNING:
        x, y = missclass_pruning(x, y, continuous=False)
    elif params[INPUT_PRUNING] == CONTINUOUS_MISSCLASSIFICATION_PRUNING:
        x, y = missclass_pruning(x, y, continuous=True)

    if params[ATTRIBUTE_SELECTION] == MODE_ABSOLUTE_VARIATION:
        find_attribute = find_by_absolut_variation
    elif params[ATTRIBUTE_SELECTION] == MODE_MISSCLASSIFICATION:
        find_attribute = find_by_missclassificaiton
    else:
        find_attribute = None
    # get output
    rt = RuleTree(class_num, attr_num)
    rt.root = build_node(x, y,
                         np.empty((0, attr_num)), np.empty((0, class_num)),
                         0, init_restrictions)
    rt.type = "DT build by ANN-DT"
    rt = rt.delete_redundancy()
    inf[TIME] = time.time()-tic
    params[INF] = inf
    return rt

