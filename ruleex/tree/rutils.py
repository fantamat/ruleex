import re
import numpy as np

try:
    import rpy2
except:
    raise ImportError("Module ruleex.tree.rutils requires rpy2!")

from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects.numpy2ri import numpy2ri
import rpy2.rlike.container as rlc

from ruleex.tree.rule import AxisRule, Leaf
from ruleex.tree.ruletree import RuleTree

def train_C5_0(x: np.array, y: np.array) -> RuleTree:
    """
    Trains a DT by C5.0 algorithm using its autor's implementaion in R lanquage
    :param x: inputs
    :param y: a list of labels
    :return: RuleTree constructed by rC50_to_ruletree function
    """
    C50 = importr('C50')
    C5_0 = robjects.r('C5.0')
    df = rlc.OrdDict()
    for i,_ in enumerate(x[0]):
        df[i] = numpy2ri(x[:,i])
    df = robjects.DataFrame(df)
    y = robjects.FactorVector(numpy2ri(y))
    c5tree_obj = C5_0(df, y)
    return rC50_to_ruletree(str(c5tree_obj[13]), ["{}".format(i) for i in range(max(y)+1)], ["X{}".format(i) for i in range(len(x[0]))])



def rC50_to_ruletree(rC50_tree: str, classes: list = None, attributes: list = None) -> RuleTree:
    """
    :param rC50_tree: string received from r command tree_model$tree where tree model was created by C5.0 function from C50 package.
        Attributes and classes must be integer values or list of those values must be passed
    example:
        id="See5/C5.0 2.07 GPL Edition 2019-02-10"
        entries="1"
        type="2" class="setosa" freq="50,50,50" att="Petal.Length" forks="3" cut="1.9"
        type="0" class="setosa"
        type="0" class="setosa" freq="50,0,0"
        type="2" class="versicolor" freq="0,50,50" att="Petal.Width" forks="3" cut="1.7"
        type="0" class="versicolor"
        type="2" class="versicolor" freq="0,49,5" att="Petal.Length" forks="3" cut="4.9000001"
        type="0" class="versicolor"
        type="0" class="versicolor" freq="0,47,1"
        type="0" class="virginica" freq="0,2,4"
    :param classes: list of names of classes (classes will be converted in to 0, 1, 2 ... as is their order in the list)
    :param attributes: list of names of attributes (order defines input values)
    :return: RuleTree whitch is copy of tree created by r
    """
    if classes:
        classes_dict = dict()
        for i, c in enumerate(classes):
            classes_dict[c] = i

    def convert_class(class_str):
        if classes:
            return classes_dict[class_str]
        else:
            return int(class_str)

    if attributes:
        attributes_dict = dict()
        for i, a in enumerate(attributes):
            attributes_dict[a] = i

    def convert_att(att_str):
        if attributes:
            return attributes_dict[att_str]
        else:
            return int(att_str)

    split_line = re.compile("(=| )")
    def parse_line(line):
        # return dictionary with values on the line
        out = dict()
        sp = split_line.split(line)
        for i, s in enumerate(sp):
            if s == "=":
                out[sp[i - 1]] = sp[i + 1].replace('"', "")
        return out

    lines = rC50_tree.split("\n")
    lines = [parse_line(line) for line in lines]
    if not lines[0]["id"].startswith("See5/C5.0"):
        print("WARNING: Tree structure is not supported.")
    if lines[1]["entries"] != "1":
        print("ERROR: There has to bee only one entry.")
        return

    inner_node_type = "2"
    leaf_node_type = "0"
    line_index = 2
    line_dict = lines[line_index]
    prev_is_inner = False # skip lines without freq
    prev_node = None
    root = None
    rule_stack = list()
    while "type" in line_dict:
        print(rule_stack)
        if line_dict["type"] == inner_node_type:
            # check correctness
            if line_dict["forks"] != "3":
                print("ERROR: Inner node (type {}) must have 3 forks!".format(inner_node_type))
                return
            # create a new node
            new_rule = AxisRule(convert_att(line_dict["att"]), float(line_dict["cut"]))
            new_rule.class_hits = eval("["+line_dict["freq"]+"]")
            if rule_stack and rule_stack[-1].false_branch:
                rule_stack[-1].true_branch = new_rule
            elif rule_stack:
                rule_stack[-1].false_branch = new_rule
            else:
                root = new_rule
            rule_stack.append(new_rule)
            prev_is_inner = True
        if line_dict["type"] == leaf_node_type:
            if not prev_is_inner:
                new_rule = Leaf({convert_class(line_dict["class"])})
                new_rule.class_hits = eval("["+line_dict["freq"]+"]")
                old_rule = rule_stack.pop()
                if old_rule.false_branch:
                    old_rule.true_branch = new_rule
                else:
                    old_rule.false_branch = new_rule
                    rule_stack.append(old_rule)
            prev_is_inner = False

        line_index += 1
        line_dict = lines[line_index]

    if classes:
        num_classes = len(classes)
    else:
        num_classes = 0
        for line_dict in lines:
            num_classes = max(num_classes, convert_class(line_dict["clss"]) + 1)

    if attributes:
        input_size = len(attributes)
    else:
        input_size = 0
        for line_dict in lines:
            input_size = max(input_size, convert_att(line_dict["att"]) + 1)

    result = RuleTree(num_classes, input_size)
    result.root = root
    return result

