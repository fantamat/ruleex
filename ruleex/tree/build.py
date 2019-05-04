import numpy as np

from ruleex.tree.ruletree import RuleTree
from ruleex.tree.rule import Leaf


def entropy(l0, l1):
    """
    computes entropy impurity for two list of labels l0 and l1
    :param l0: list of labels
    :param l1: list of labels
    :return: entropy impurity
    """
    _, p0 = np.unique(l0, return_counts=True)
    p0 = p0 / np.sum(p0)
    _, p1 = np.unique(l1, return_counts=True)
    p1 = p1 / np.sum(p1)
    return - np.sum(p0 * np.log2(p0)) - np.sum(p1 * np.log2(p1))


def gini(l0, l1):
    """
    computes gini impurity for two list of labels l0 and l1
    :param l0: list of labels
    :param l1: list of labels
    :return: gini impurity
    """
    _, p0 = np.unique(l0, return_counts=True)
    p0 = p0 / np.sum(p0)
    _, p1 = np.unique(l1, return_counts=True)
    p1 = p1 / np.sum(p1)
    return np.sum(p0 * (1 - p0)) + np.sum(p1 * (1 - p1))


def get_leaf(labels, class_num):
    """
    creates tŕee.Leaf object, i.e., subclass of tree.Rule
    class_set will be set to maximal occurring class in labels
    :param labels: a list of labels
    :param class_num: the number of classes
    :return: tree.Leaf with maximal occurring class as class_set
    """
    values, counts = np.unique(labels, return_counts=True)
    c = np.zeros(class_num)
    c[values] = counts
    max_label = np.argmax(c)
    node = Leaf({max_label})
    node.class_hits = c
    return node


def build_from_rules(rule_list, data, labels,
                     min_split_fraction: float = 0.98,
                     num_of_classes=None,
                     impurity=gini,
                     max_depth=None):
    """
    Builds tree.RuleTree from available splits from rule_list using data and its labels
    :param rule_list: a list of rules (subclass of tree.Rule)
    :param data: a list of inputs of the samples
    :param labels: a list of classificaitons for the samples
    :param min_split_fraction: minimal split fraction
    :param num_of_classes: number of classes
    :param impurity: impurity measure (function which takes two labels and return impurity value)
    :param max_depth: maximal depth of built tree
    :return: built RuleTree
    """
    if not num_of_classes:
        num_of_classes = np.max(labels)+1

    def build_from_rules_rek(rule_list, x, l, depth):
        if max_depth is not None and depth == max_depth:
            return get_leaf(l, num_of_classes)
        best_impurity = np.inf
        best_node_index = None
        best_node_mask = None
        for i, node in enumerate(rule_list):
            mask = node.eval_all(x)
            if np.all(mask) or not np.any(mask):
                continue # node do not provide any split for given data
            if np.mean(mask) > min_split_fraction or np.mean(mask) < 1 - min_split_fraction:
                continue
            node_imp = impurity(l[mask], l[~mask])
            if node_imp < best_impurity:
                best_impurity = node_imp
                best_node_index = i
                best_node_mask = mask
        if best_node_index is None:
            return get_leaf(l, num_of_classes)
        node = rule_list[best_node_index].copy()
        next_rule_list = rule_list.copy()
        next_rule_list.pop(best_node_index)

        node.true_branch = build_from_rules_rek(next_rule_list, x[best_node_mask], l[best_node_mask], depth + 1)
        node.false_branch = build_from_rules_rek(next_rule_list, x[~best_node_mask], l[~best_node_mask], depth + 1)
        return node

    rt = RuleTree(num_of_classes, data.shape[1])
    rt.root = build_from_rules_rek(rule_list, data, labels, 0)
    return rt

