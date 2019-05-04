import numpy as np
from sklearn.model_selection import KFold

from ruleex.tree.rule import Leaf
from ruleex.tree.ruletree import RuleTree


def prune_cost_complex(ruletree, x, y, folds=5):
    """
    WARNING: NOT TESTED!
    Cost-complexity pruning method
    First the values of alpha are computed on all data
    then cross validation is used to select subtrees T_s with minimal R_alpha(T_s) and computing this tree accuracy.
    The alpha with highest mean accuracy is choossen and tree from the first stem for this alpha is returned
    Node:
        R_alpha(T) = R(T) + alpha * #leafs
        R(T) = sum( [ (1-max(p_j)) * #samples_in_t / #samples for t as leaf_nodes of T ] )
    Warning:
        if the variance of choosen alpha from cross-validation is higher than 0.1 the warning is printed
    :param x: input data
    :param y: output labels
    :param folds: number of folds used in cross-validation
    :param accuracy_drop:
    :return: self (pruned tree)
    """

    _rt = RuleTree(ruletree.num_of_classes, ruletree.input_size)

    def get_all_hits(nodes):
        out = list()
        for node in nodes:
            out.append(node.class_hitsf)
        return out

    def get_R(node_hits, num_samples):
        return (np.sum(node_hits) - np.max(node_hits))/num_samples

    def get_alpha(node, hits, num_samples):
        R_t = get_R(hits[node])
        _rt.root = node
        new_nodes = _rt.get_all_nodes()
        R_T_t = .0
        T_t = 0
        for new_node in new_nodes:
            if isinstance(new_node, Leaf):
                T_t += 1
                R_T_t += get_R(hits[new_node])
        return (R_t - R_T_t) / (T_t - 1), new_nodes

    def R_alpha_T_decrease(node, alpha, hits, fold, num_samples):
        # node is inner node with two leafs
        # returns R_alpha(T)-R_alpha(T_node) > 0
        # R_alpha(T) = R(T) + alpha * |T_leafs|
        # if node will be leaf then T_leafs drop by 1 and R(T) - R(T_new) = R(node_false) + R(node_true) - R(node)
        # R_alpha(T)-R_alpha(T_node) = R(node_false) + R(node_true) - R(node) + alpha
        R_diff = get_R(hits[node.false_branch][fold], num_samples) \
                    + get_R(hits[node.true_branch][fold], num_samples) \
                    - get_R(hits[node][fold], num_samples)
        return R_diff + alpha > 0

    if len(x) < 10*folds:
        print("ERROR: Cost-complexity pruning uses cross-validation for input data so it needs at least 10 times "
              "number of used folds." 
              "\n       Called with {} samples and {} folds (5 is default).".format(len(x), folds)
              )
        return
    all_nodes = ruletree.get_all_nodes()
    if ruletree.graph_max_indegree(all_nodes=all_nodes) > 1:
        print("ERROR: Decision tree graph must have indegree 1 for all vertexes. Cost-complexity pruning failed.")
        return
    ruletree.fill_hits(x, y)
    all_hits_dict = dict((node, hit) for node, hit in zip(all_nodes, get_all_hits(all_nodes)))
    leaf_set = set()  # leaf set defines subtree
    for node in all_nodes:
        if isinstance(node, Leaf):
            leaf_set.add(node)
    primal_leaf_sets = [leaf_set]
    alpha_list = list() # len(alpha_list) + 1 == len(primal_lef_sets)
    # find all subtrees defined by primal_leaf_sets with minimal alpha
    while primal_leaf_sets[-1] != all_nodes:
        min_alpha = None
        min_alpha_node = None
        new_leafs = None
        for node in all_nodes.difference(primal_leaf_sets[-1]):
            alpha, nodes = get_alpha(node, all_hits_dict, len(x))
            if not min_alpha_node or min_alpha > alpha:
                min_alpha = alpha
                min_alpha_node = node
                new_leafs = nodes
        primal_leaf_sets.append(primal_leaf_sets[-1].union(new_leafs))
        alpha_list.append(min_alpha)

    cv_hits_train = list()
    cv_hits_test = list()
    train_fold_samples = list()
    test_fold_samples = list()
    kf = KFold(n_splits=folds)
    for train_i, test_i in kf.split(x, y):
        train_fold_samples.append(len(train_i))
        test_fold_samples.append(len(test_i))
        ruletree.fill_hits(x[train_i], y[train_i])
        cv_hits_train.append(get_all_hits(all_nodes))
        ruletree.fill_hits(x[test_i], y[test_i])
        cv_hits_test.append(get_all_hits(all_nodes))
    cv_hits_train = np.array(cv_hits_train)
    cv_hits_test = np.array(cv_hits_test)
    train_hits_dict = dict()
    test_hits_dict = dict()
    for i, node in enumerate(all_nodes):
        train_hits_dict[node] =  cv_hits_train[:, i, :]
        test_hits_dict[node] = cv_hits_test[:, i, :]

    accuracy_all = np.zeros((len(alpha_list), folds))
    for i, alpha in enumerate(alpha_list):
        for fold in range(folds):
            ls = leaf_set.copy()
            changed = True
            # find tree T with minimal R_alpha(T)
            while changed:
                changed = False
                for node in all_nodes:
                    if node.true_branch in ls and node.false_branch in ls and \
                       R_alpha_T_decrease(node, alpha, train_hits_dict, fold, train_fold_samples[fold]):
                        ls.add(node)
                        changed = True
            # compute accuracy for tree T
            accuracy = .0
            for inner_node in all_nodes.difference(ls):
                if node.true_branch in ls:
                    accuracy += get_R(test_hits_dict[node.true_branch][fold], test_fold_samples[fold])
                if node.false_branch in ls:
                    accuracy += get_R(test_hits_dict[node.false_branch][fold], test_fold_samples[fold])
            accuracy_all[i, fold] = accuracy
    accuracy_mean = np.mean(accuracy_all, axis=1)
    best_alpha_i = np.argmax(accuracy_mean)

    accuracy_var = np.var(accuracy_all, axis=1)
    # check variance of accuracy and print warning
    if accuracy_var[best_alpha_i] > 0.1:
        print("WARNING: Const-complexity prunning inner cross-validation variance of accuracy is high {}."
              .format(accuracy_var[best_alpha_i]))

    best_leaf_set = primal_leaf_sets[best_alpha_i+1]
    # replace nodes
    for inner_node in all_nodes.difference(best_leaf_set):
        if inner_node.true_branch in best_leaf_set and not isinstance(inner_node.true_branch, Leaf):
            inner_node.true_branch = Leaf({int(np.argmax(all_hits_dict[inner_node.true_branch]))})
        if inner_node.false_branch in best_leaf_set and not isinstance(inner_node.false_branch, Leaf):
            inner_node.false_branch = Leaf({int(np.argmax(all_hits_dict[inner_node.false_branch]))})
    return ruletree

