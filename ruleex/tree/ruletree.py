import os
import pickle

from ruleex.tree.rule import Leaf, AxisRule, Rule, LinearRule
from graphviz import Digraph
import numpy as np


class RuleTree:
    """
    Representation of the decision tree (DT) or more generally a decision directed acyclic graph (DDAG)
    It do not handle the maitanance of the graph but it implements the functions that are performed on its structure
    The user is responsible for the attributes and its handling
        The usage is harder. However, it allows user to optimize algorithms made on the graph
    """
    __slots__ = ("root",
                 "terms",
                 "num_of_classes",
                 "input_size",
                 "type",
                 "class_description",
                 "num_of_nodes",
                 "node_set")

    @staticmethod
    def half_in_ruletree(number_of_classes):
        """
        Creates a DT with the form
            if x(0) > 0.5 then class=0
            else
                if x(1) > 0.5 then class=1
                else
                    if x(2) > 0.5 then class=2
                    ...
                    else class=number_of_classes-1
        :param number_of_classes: the number of output classes
        :return: constructed RuleTree
        """
        output = RuleTree(number_of_classes, number_of_classes)
        for i in range(number_of_classes-1):
            rule = AxisRule(i, 0.5, trueBranch=Leaf([i]), classSet=[j for j in range(i, number_of_classes)])
            if i == 0:
                output.root = rule
                old_rule = rule
            else:
                old_rule.false_branch = rule
                old_rule = rule
        old_rule.false_branch = Leaf({number_of_classes-1})

        return output

    def __init__(self, num_class, input_size):
        """
        Creates a RuleTree with Leaf with all classes in the root
        :param num_class: the number of classes
        :param input_size: the size of the input
        """
        self.root = Leaf(list(range(num_class)))
        self.num_of_classes = num_class
        self.input_size = input_size
        self.type = "undefined"
        self.class_description = dict()

    def eval_one(self, x, return_last_rule=False):
        """
        Evaluates one sample x
        :param x: the input value
        :param return_last_rule: a flag
        :return: output of the last rule in the RuleTree structure for given samples
            if return_last_rule is True then returns also the last rule in the RuleTree structure
        """
        act_result = None
        act_rule = self.root
        while act_rule:
            act_result = act_rule.class_set
            last_rule = act_rule
            act_rule = act_rule.get_next(x)
        # check
        if max(act_result) >= self.num_of_classes:
            print("Warning: returning class_set with higher index of class than expected!")
        if return_last_rule:
            return act_result, last_rule
        return act_result

    def eval_all(self, x, all_nodes=None, prevs=None):
        """
        Evaluates all samples in the list x
        :param x: a list of the input of the samples
        :param all_nodes: all node of the RuleTree if defines then the process is speed up
        :param prevs: a dictionary that maps a (Rule, bool) to the next node in the RuleTree structure
            if defines then the process is speed up
        :return: a list of the first classes in the Leaf's class_set to which the samples falls
        """
        all_nodes = self.get_all_nodes()
        prevs = self.get_predecessor_dict(all_nodes)
        if all_nodes and prevs:
            # fast mode
            if isinstance(self.root, Leaf):
                return np.array(self.root.eval_all(x))

            occupied = {self.root:x}
            occupied_index = {self.root:np.arange(len(x))}
            leaf_results_index = dict()
            prevs = prevs.copy() # make local copy of prevs
            for node in prevs.keys():
                s = set()
                for p,branch in prevs[node]:
                    s.add(p)
                prevs[node] = s
            prevs[self.root] = {}
            while occupied:
                act_node = None
                for enode in occupied.keys():
                    if not prevs[enode]:
                        act_node = enode
                        break
                if not act_node:
                    break
                o = act_node.eval_all(occupied[act_node]) # obtain filter for decision np boolean array
                tx, ti = occupied[act_node][o], occupied_index[act_node][o]
                tn = act_node.true_branch
                fx, fi = occupied[act_node][~o], occupied_index[act_node][~o]
                fn = act_node.false_branch
                if act_node in prevs[tn]:
                    prevs[tn].remove(act_node)
                if isinstance(tn, Leaf):
                    if tn in leaf_results_index:
                        leaf_results_index[tn] = np.concatenate((leaf_results_index[tn], ti), axis=0)
                    else:
                        leaf_results_index[tn] = ti
                else:
                    if tn in occupied:
                        occupied[tn] = np.concatenate((occupied[tn], tx), axis=0)
                        occupied_index[tn] = np.concatenate((occupied_index[tn], ti), axis=0)
                    else:
                        occupied[tn] = tx
                        occupied_index[tn] = ti
                if act_node in prevs[fn]:
                    prevs[fn].remove(act_node)
                if isinstance(fn, Leaf):
                    if fn in leaf_results_index:
                        leaf_results_index[fn] = np.concatenate((leaf_results_index[fn], fi), axis=0)
                    else:
                        leaf_results_index[fn] = fi
                else:
                    if fn in occupied:
                        occupied[fn] = np.concatenate((occupied[fn], fx), axis=0)
                        occupied_index[fn] = np.concatenate((occupied_index[fn], fi), axis=0)
                    else:
                        occupied[fn] = fx
                        occupied_index[fn] = fi
                occupied.pop(act_node)
                occupied_index.pop(act_node)
            out = np.zeros(len(x), dtype=np.int)
            for leaf in leaf_results_index.keys():
                if len(leaf.class_set) > 1:
                    raise ValueError("In fast mode of ruletree.eval_all all leafs need to have one item in class set!")
                out[leaf_results_index[leaf]] = list(leaf.class_set)[0]
            return out
        else:
            # slow mode
            out = list()
            all_single_values = True
            for i in range(len(x)):
                next = self.eval_one(x[i])
                if all_single_values and len(next) != 1:
                    all_single_values = False
                out.append(next)
            if all_single_values:
                for i, v in enumerate(out):
                    out[i] = list(v)[0]
            return out

    def replace_leaf_with_set(self, leaf_class_set, replacing_rule, all_nodes=None, prevs=None):
        if not issubclass(type(replacing_rule), Rule):
            return "ERROR replacingRule must be of the Rule type!"
        if not all_nodes:
            all_nodes = self.get_all_nodes()
        if not prevs:
            prevs = self.get_predecessor_dict(all_nodes)
        for rule in all_nodes:
            if type(rule) is Leaf and rule.class_set == set(leaf_class_set):
                for prev, branch in prevs[rule]:
                    if prev == self:
                        self.root = replacing_rule
                        break
                    if branch:
                        prev.true_branch = replacing_rule
                    else:
                        prev.false_branch = replacing_rule
        return self

    def replace_rule(self, matching_rule, replacement, pred_entries=None):
        """
        Replace the rule matching_rule by the replacement
        :param matching_rule: a Rule object that is part of the RuleTree structure
        :param replacement: a Rule object
        :param pred_entries: a dictionary that maps a (Rule, bool) to the next node in the RuleTree structure
            if defines then the process is speed up
        :return: self
        """
        def make_replacement(pred_entries, act_rule):
            if act_rule == matching_rule:
                # make raplacement
                for pred, branch in pred_entries:
                    if type(pred) is RuleTree:
                        pred.root = replacement
                    else:
                        if branch:
                            pred.true_branch = replacement
                        else:
                            pred.false_branch = replacement
                return True
            else:
                return False

        if pred_entries:
            make_replacement(pred_entries, matching_rule)
        else:
            preds = self.get_predecessor_dict()
            unvisited_rules = [self.root]
            while len(unvisited_rules) > 0:
                rule = unvisited_rules[-1]
                del unvisited_rules[-1]
                if make_replacement(preds[rule], rule):
                    break
                else:
                    if rule.true_branch:
                        unvisited_rules.append(rule)
                    if rule.false_branch:
                        unvisited_rules.append(rule)
        return self

    def to_string(self):
        """
        Print a RuleTree structure to the string
        :return:
        """
        tab_constant = "\t"
        tabs = ""

        def rek_to_string(rule, tab):
            out = tab + "if (" + rule.to_string() + ") {\n"
            if not rule.true_branch and not rule.false_branch:
                return tab + str(rule.class_set)
            elif not rule.true_branch or not rule.false_branch:
                if rule.true_branch:
                    r = rule.true_branch
                else:
                    r = rule.false_branch
                out += tab + rek_to_string(r, tab + "tab_constant")
            else:
                if type(rule.true_branch) is Leaf:
                    out += tab + tab_constant + rule.true_branch.to_string()
                else:
                    out += rek_to_string(rule.true_branch, tab + tab_constant)
                out += "\n" + tab + "} else {\n"
                if type(rule.false_branch) is Leaf:
                    out += tab + tab_constant + rule.false_branch.to_string()
                else:
                    out += rek_to_string(rule.false_branch, tab + tab_constant)
            out += "\n" + tab + "}"
            return out

        return rek_to_string(self.root, tabs)

    def view_graph(self, filename=None, varbose=True, true_string="true", false_string="false", all_nodes=None):
        """
        Displays the RuleTree structure as graphivz's Digraph
        :param filename: a finename to which is save gprahivz model and pdf that is generated if the varbose is True
        :param varbose: if True then the result is viewed
        :param true_string: a string that shold be printed as the description of the true_branch edges
        :param false_string: a string that shold be printed as the description of the false_branch edges
        :param all_nodes: all Rule objects that occur in the RuleTree structure
        :return: representation of the RuleTree by a graphviz Digraph
        """
        g = Digraph(name="RuleTree", filename=filename)

        def rek_view_graph(rule, edges, nodes):
            if not rule.true_branch and not rule.false_branch:
                pass  # nothing
            else:
                if rule.true_branch:
                    if not rule.true_branch in nodes:
                        g.node(str(rule.true_branch.__hash__()), label=rule.true_branch.to_string(True))
                        nodes.add(rule.true_branch)
                    if not (rule, True, rule.true_branch) in edges:
                        g.edge(str(rule.__hash__()), str(rule.true_branch.__hash__()), label=true_string)
                        edges.add((rule, True, rule.true_branch))
                if rule.false_branch:
                    if not rule.false_branch in nodes:
                        g.node(str(rule.false_branch.__hash__()), label=rule.false_branch.to_string(True))
                        nodes.add(rule.false_branch)
                    if not (rule, False, rule.false_branch) in edges:
                        g.edge(str(rule.__hash__()), str(rule.false_branch.__hash__()), label=false_string)
                        edges.add((rule, False, rule.false_branch))

        g.node(str(self.root.__hash__()), label=self.root.to_string(True))
        if not all_nodes:
            all_nodes = self.get_all_nodes()
        ed = set()
        no = set()
        for rule in all_nodes:
            rek_view_graph(rule, ed, no)
        if varbose:
            g.view()
        return g

    def copy(self):
        """
        :return: a copy of the Ruletree and all its nodes
        """
        output = RuleTree(self.num_of_classes, self.input_size)
        output.root = self.root.copy()
        return output

    def save(self, filename="rule_tree.pic"):
        """
        Saves the RuleTree as binary object - using pickle
        :param filename: a fileneme to which the RuleTree will be saved
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename="rule_tree.pic"):
        """
        Load the RuleTree from the file with filename saved by the method RuleTree.save
        :param filename: a name of the file to which the RuleTree was saved by pickle
        :return: a loaded RuleTree
        """
        with open(filename, "rb") as file:
            output = pickle.load(file)
        return output

    def get_all_nodes(self):
        """
        :return: all current nodes of the RuleTree structure
        """
        self.node_set = set()
        rec_set = set([self.root])
        while len(rec_set) > 0:
            rule = rec_set.pop()
            if rule in self.node_set:
                continue
            self.node_set.add(rule)
            if rule.true_branch:
                rec_set.add(rule.true_branch)
            if rule.false_branch:
                rec_set.add(rule.false_branch)
        self.num_of_nodes = len(self.node_set)
        return self.node_set

    def get_all_expanded_rules(self):
        """
        :return: a list of rules which are represented by a list of Rule object along the path to the leaf
            of the RuleTree structure
        """
        def get_all_expanded_rules_rek(rule, act_expanded_rule, expanded_rules):
            if type(rule) is Leaf:
                expanded_rules.append(act_expanded_rule + [rule])
            else:
                if rule.false_branch:
                    get_all_expanded_rules_rek(rule.false_branch, act_expanded_rule + [(False, rule)], expanded_rules)
                if rule.true_branch:
                    get_all_expanded_rules_rek(rule.true_branch, act_expanded_rule + [(True, rule)], expanded_rules)

        output = list()
        get_all_expanded_rules_rek(self.root, [], output)
        return output

    def get_predecessor_dict(self, all_nodes=None):
        """
        :param all_nodes: all node of the RuleTree if defines then the process is speed up
        :return: a dictionary that maps a (Rule, bool) to the next node in the RuleTree structure
        """
        if not all_nodes:
            all_nodes = self.get_all_nodes()
        output = dict()
        output[self.root] = [(self, True)]
        for rule in all_nodes:
            if rule.true_branch:
                if rule.true_branch in output:
                    output[rule.true_branch].append((rule, True))
                else:
                    output[rule.true_branch] = [(rule, True)]
            if rule.false_branch:
                if rule.false_branch in output:
                    output[rule.false_branch].append((rule, False))
                else:
                    output[rule.false_branch] = [(rule, False)]
        return output

    def print_expanded_rules(self):
        """
        Print all rules stored in the RuleTree structure to the standard output
        Warning: It can lead to very long description so it is conviniente to check the number of rule first
            by len(rt.get_all_expanded_rules())
        """
        er = self.get_all_expanded_rules()
        for rl in er:
            str = "("
            for i, r in enumerate(rl):
                if i < len(rl) - 1:
                    str += (r.to_string())
                if i < len(rl) - 2:
                    str += (") and (")
                if i == len(rl) - 1:
                    str += (") then ")
                    str += (r.to_string())
            print(str + "\n")

    def check_none_inside_tree(self):
        """
        Checks whater the RuleTree structure is valid, i.e., all paths ends with the Leaf object
        :return: True if the RuleTree is valid
        """
        def check_none_in_tree_rek(rule):
            if not type(rule) is Leaf:
                if rule:
                    return check_none_in_tree_rek(rule.false_branch) or check_none_in_tree_rek(rule.true_branch)
                else:
                    return True
            else:
                return False

        return check_none_in_tree_rek(self.root)

    def check_one_evaluation(self):
        """
        Checks if all Leafs of the RuleTree structure have only one class in their class_set
        """
        def check_one_evaluation_rek(rule):
            if not type(rule) is Leaf:
                if rule:
                    return check_one_evaluation_rek(rule.false_branch) and check_one_evaluation_rek(rule.true_branch)
                else:
                    return True
            else:
                return len(rule.class_set) == 1

        return check_one_evaluation_rek(self.root)

    def graph_max_indegree(self, all_nodes=None, return_all=False):
        """
        Computes the maximal indegree of the RuleTree structure (outdegree is alwas 0 for Leaf and 2 for other nodes)
        :param all_nodes:
        :param return_all: a flag
        :return:
            if return_all is True then a dictionary of indegrees for all nodes in the RuleTree structure
            otherwise a maximal indegree that is found in the RuleTree structure
        """
        indegrees = self.get_predecessor_dict(all_nodes=all_nodes)
        for node in indegrees.keys():
            indegrees[node] = len(indegrees[node])
        if return_all:
            return indegrees
        else:
            return max(indegrees.values())

    def fill_class_sets(self):
        """
        Fills class sets inside the tree structure. Uses leaf class sets and then union to generate
        class sets inside the tree.
        """

        def fill_class_set_by_leafs_rek(rule, class_set_pointer):
            if not type(rule) is Leaf:
                if rule:
                    false_class_set = class_set_pointer.copy()
                    true_class_set = class_set_pointer.copy()
                    fill_class_set_by_leafs_rek(rule.false_branch, false_class_set)
                    class_set_pointer[0] = class_set_pointer[0].union(false_class_set[0])
                    fill_class_set_by_leafs_rek(rule.true_branch, true_class_set)
                    class_set_pointer[0] = class_set_pointer[0].union(true_class_set[0])
                    rule.class_set = class_set_pointer[0]
                else:
                    return
            else:
                class_set_pointer[0] = class_set_pointer[0].union(rule.class_set)

        fill_class_set_by_leafs_rek(self.root, [set()])

    def get_thresholds(self, all_nodes=None):
        """
        Goes through the graph and return all pair index and threshold of the AxisRule
        that occur in the RuleTree structure
        :param all_nodes:
        :return: a list of pair (index, threshold)
        """
        if not all_nodes:
            all_nodes = self.get_all_nodes()
        out = set()
        for node in all_nodes:
            if isinstance(node, AxisRule):
                out.add((node.i, node.b))
        return list(out)

    def delete_redundancy(self):
        """
        Deletes all occurring redundancy (currently only calls delte_class_redundancy method)
        :return: self
        """
        #self.delete_interval_redundancy()
        self.delete_class_redundancy()
        return self

    def delete_interval_redundancy(self):
        """
        Deletes the rules which result is forced by previous nodes in the path to it.
        :return: self with reduced structure
        """
        print("[ruletree]: Warning delete interval redundancy is implementet only for tree structure graphs.")
        def deleteRedundanciRek(pred, rule, intervals):
            if type(rule) is AxisRule:
                if rule.b <= intervals[rule.i][1]:
                    if rule.b > intervals[rule.i][0]:
                        oldValue = intervals[rule.i][1]
                        intervals[rule.i][1] = rule.b
                        deleteRedundanciRek(rule, rule.false_branch, intervals)
                        intervals[rule.i][1] = oldValue
                    else:
                        if pred.true_branch == rule:
                            pred.true_branch = rule.true_branch
                            deleteRedundanciRek(pred, rule.true_branch, intervals)
                        else:
                            pred.false_branch = rule.true_branch
                            deleteRedundanciRek(pred, rule.true_branch, intervals)
                        return
                else:
                    # redundant rule always true
                    if type(pred) is RuleTree:  # is a root
                        pass
                    else:
                        if pred.true_branch == rule:
                            pred.true_branch = rule.false_branch
                            deleteRedundanciRek(pred, rule.false_branch, intervals)
                        else:
                            pred.false_branch = rule.false_branch
                            deleteRedundanciRek(pred, rule.false_branch, intervals)
                        return

                if rule.b > intervals[rule.i][0]:
                    if rule.b <= intervals[rule.i][1]:
                        oldValue = intervals[rule.i][0]
                        intervals[rule.i][0] = rule.b
                        deleteRedundanciRek(rule, rule.true_branch, intervals)
                        intervals[rule.i][0] = oldValue
                    else:
                        if pred.true_branch == rule:
                            pred.true_branch = rule.false_branch
                            deleteRedundanciRek(pred, rule.false_branch, intervals)
                        else:
                            pred.false_branch = rule.false_branch
                            deleteRedundanciRek(pred, rule.false_branch, intervals)
                        return
                else:
                    # redundant rule always true
                    if type(pred) is RuleTree:  # is a root
                        pass
                    else:
                        if pred.true_branch == rule:
                            pred.true_branch = rule.true_branch
                            deleteRedundanciRek(pred, rule.true_branch, intervals)
                        else:
                            pred.false_branch = rule.true_branch
                            deleteRedundanciRek(pred, rule.true_branch, intervals)

        ints = [[-np.inf, np.inf] for i in range(self.input_size)]
        deleteRedundanciRek(self, self.root, ints)
        return self

    def delete_class_redundancy(self):
        """
        Deletes the redundancy of parts of the RuleTree structure which lead to the same result
        :return: self without redundant nodes
        """
        def delete_redundancy_rek(prev, rule):
            if type(rule) is Leaf:
                return rule.class_set
            else:
                true_class_set = delete_redundancy_rek(rule, rule.true_branch)
                false_class_set = delete_redundancy_rek(rule, rule.false_branch)
                rule.class_set = true_class_set | false_class_set
                if rule.true_branch == rule.false_branch or (type(rule.true_branch) is Leaf and type(
                        rule.false_branch) is Leaf and true_class_set == false_class_set):
                    # delete act rule
                    if isinstance(prev, Rule):
                        if prev.true_branch == rule:
                            prev.true_branch = rule.true_branch
                            del rule
                        else:
                            prev.false_branch = rule.false_branch
                            del rule
                return true_class_set | false_class_set

        delete_redundancy_rek(self, self.root)
        return self

    def fill_hits(self, x, labels, remove_redundant=False):
        """
        Fills class_hits on each Rule in the RuleTree structure
        :param x: a list of inputs
        :param labels: a list of labels for the inputs
        :return: self
        """
        visited_rules = set()
        for i, label in enumerate(labels):
            actRule = self.root
            while actRule:
                if actRule not in visited_rules:
                    actRule.class_hits = [0] * self.num_of_classes
                    visited_rules.add(actRule)
                actRule.class_hits[label] += 1
                actRule = actRule.get_next(x[i])
        if remove_redundant:
            self.remove_unvisited(visited_rules)
        return self

    def remove_unvisited(self, visited_rules):
        """
        Removes unvisited nodes of the RuleTree structure
        :param visited_rules: a list of Rules of the RuleTree structure that was visited
        :return: self with removed rules that are not in visited_rules
        """
        # choose new root
        def remove_redundant_rek(prev, rule):
            new_root = rule
            while not isinstance(new_root, Leaf) and \
                    ((new_root.false_branch not in visited_rules) or (new_root.true_branch not in visited_rules)):
                if new_root.false_branch not in visited_rules:
                    new_root = new_root.true_branch
                    break
                if new_root.true_branch not in visited_rules:
                    new_root = new_root.false_branch
            if prev == self:
                self.root = new_root
            else:
                if prev.true_branch == rule:
                    prev.true_branch = new_root
                else:
                    prev.false_branch = new_root

        rek_set = [(self, self.root)]
        while len(rek_set) > 0:
            prev, rule = rek_set[-1]
            del rek_set[-1]
            remove_redundant_rek(prev, rule)
            if rule.true_branch and rule.true_branch.true_branch != rule.true_branch.false_branch:
                rek_set.append((rule, rule.true_branch))
            if rule.false_branch and rule.false_branch.true_branch != rule.false_branch.false_branch:
                rek_set.append((rule, rule.false_branch))
        return self


    def delete_node(self, node, branch, all_nodes, prevs):
        """
        Delete node and substitute it by its true of false branch (the variable brach defines the behaviour)
        :param node: a node to delete
        :param branch: a branch that is used instead of the deleted node
        :param all_nodes:
        :param prevs:
        """
        all_nodes.remove(node)
        if branch:
            new_node = node.true_branch
        else:
            new_node = node.false_branch
        if node == self.root:
            self.root = new_node
            del prevs[new_node]
        else:
            prevs[new_node].remove((node, branch))
            for prev, br in prevs[node]:
                prevs[new_node].append((prev, br))
                if br:
                    prev.true_branch = new_node
                else:
                    prev.false_branch = new_node

    def remove_unused_edges(self, all_nodes, prevs):
        """
        Removal of the edges that are not used by the sample in the evaluation.
        It also removes duplicate eges (true_branch == false_branch)
        Changes are also propagated into list all_nodes and dict prevs
        :param all_nodes:
        :param prevs:
        :return: self
        """
        processed = set()
        all_nodes_local = all_nodes.copy()
        for act_node in all_nodes_local:
            if act_node in processed or not act_node:
                continue
            processed.add(act_node)
            if act_node.true_branch and (act_node.true_branch == act_node.false_branch):
                self.delete_node(act_node, True, all_nodes, prevs)
                continue
            if act_node.num_true or act_node.num_false:
                if act_node.num_true == 0:
                    self.delete_node(act_node, False, all_nodes, prevs)
                    continue
                if act_node.num_false == 0:
                    self.delete_node(act_node, True, all_nodes, prevs)
                    continue
        return self

    def get_rule_index_dict(self, x):
        """
        :param x: a list of the inputs
        :return: a dictionary in which keys are rules of the tree and values are lists of indexes of x that visited rules during evaluation
        """
        output = dict()
        for i, xx in enumerate(x):
            act_rule = self.root
            while act_rule:
                if act_rule not in output:
                    output[act_rule] = list()
                output[act_rule].append(i)
                act_rule = act_rule.get_next(x[i])
        return output

    def prune_using_hits(self, min_split_fraction=0.0, min_rule_samples=1, all_nodes=None):
        """
        Deletes the nodes that are not used by the last evaluation,
            or which divides the samples in the ratio less then min_split_fraction,
            or which evaluates less than min_rule_samples samples
        :param min_split_fraction: minimal ratio of the samples in the split node
        :param min_rule_samples: minimal number of evaluated samples on each node
        :param all_nodes:
        :return: self
        """
        if not all_nodes:
            all_nodes = self.get_all_nodes()

        def delete_rule(prevEntries, rule, rules2delete):
            new_root = rule
            while not isinstance(new_root, Leaf) and new_root in rules2delete.keys():
                # del_rule = new_root
                if rules2delete[new_root] != None:
                    if rules2delete[new_root]:
                        new_root = new_root.true_branch
                    else:
                        new_root = new_root.false_branch
                else:
                    new_root = Leaf(rule.class_set)
                    new_root.class_hits = rule.class_hits
                # del rules2delete[del_rule]
            for prev, branch in prevEntries:
                if prev == self:
                    self.root = new_root
                else:
                    if branch:
                        prev.true_branch = new_root
                    else:
                        prev.false_branch = new_root

        rules2delete = dict()
        prev = self.get_predecessor_dict(all_nodes)
        for rule in all_nodes:
            if rule.true_branch and rule.false_branch:
                if rule.class_hits:
                    hits = sum(rule.class_hits)
                else:
                    hits = 0
                if hits < min_rule_samples:
                    rules2delete[rule] = None
                    continue
                if rule.true_branch.class_hits:
                    true_hits = sum(rule.true_branch.class_hits)
                else:
                    true_hits = 0
                if rule.false_branch.class_hits:
                    false_hits = sum(rule.false_branch.class_hits)
                else:
                    false_hits = 0
                if min(true_hits, false_hits) / hits < min_split_fraction:
                    if true_hits > false_hits:
                        rules2delete[rule] = True
                    else:
                        rules2delete[rule] = False
        # while len(rules2delete) > 0:
        #    rule = list(rules2delete.keys())[0]
        for rule in rules2delete.keys():
            delete_rule(prev[rule], rule, rules2delete)
        return self

    def get_stats(self):
        """
        Return some desired properties of the ruletree. It recursively walks through the decision graph.
        :return: tuple that contains:
            rule_count - number of rules
            node_count - number of nodes
            index_count - number of used indexes by the tree (supports only AxisRule and LinearRule)
            sum_rule_len - sum of the lenght of all rules
            sum_rule_index - sum of indexes used in each rule
        """
        def process_node(rule, node_set, index_set, prev_nodes, p):
            node_set.add(rule)
            if isinstance(rule, AxisRule):
                index_set.add(rule.i)
            elif isinstance(rule, LinearRule):
                index_set = index_set | set(list(np.argwhere(rule.a != 0).T[0]))
            elif isinstance(rule, Leaf):
                p["rule_count"] += 1
                p["sum_rule_len"] += len(prev_nodes)
                s = set()
                for r in prev_nodes:
                    if isinstance(r, LinearRule):
                        s = s | set(list(np.argwhere(r.a != 0).T[0]))
                    elif isinstance(r, AxisRule):
                        s.add(r.i)
                p["sum_rule_index"] += len(s)

            prev_nodes.append(rule)
            if rule.false_branch is not None:
                process_node(rule.false_branch, node_set, index_set, prev_nodes, p)
            if rule.true_branch is not None:
                process_node(rule.true_branch, node_set, index_set, prev_nodes, p)
            del prev_nodes[-1]

        node_set = set()
        index_set = set()
        p = {
            "rule_count": 0,
            "sum_rule_len": 0,
            "sum_rule_index": 0,
        }
        process_node(self.root, node_set, index_set, prev_nodes=[], p=p)
        return p["rule_count"], len(node_set), len(index_set), p["sum_rule_len"], p["sum_rule_index"]

