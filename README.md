# ruleex

The *ruleex* is a python package which implements rule extraction algorithms.
Now, there are available three algorithms (HypInv, ANN-DT, and DeepRED).
All of those algorithms were slightly modified. Especially, extracted rules are
stored in the tree structure with decision nodes which can have any form, e.g.,
axis parallel or linear split. Implementation of the general decision tree, i.e.,
the tree with any kind of decision node, is also included along with some handy
operations.

Following description provides only short description focused on the code (classes, 
functions, and their arguments). For more details of the theory behind the implementation
there will be available my master thesis.

## tree

This subpackage contains implementation of the general decision tree, i.e.,
decision tree with any kind of decision nodes. It is also possible to operate with
so-called decision graph which can have more than one incoming edges to each node.

### class Rule

Rule is an abstract class describing the decision node inside the tree.
Object of this class stores at least these arguments:

* *true_branch and false_branch*: pointers to the next node or None.
* *class_set*: set of classes that the node represents (in the tree 
    this sets should follow the order in inclusion manner) 
* *class_hits*: the list with the number of samples from each class.
* *num_true and num_false*: the number of samples that were redirected to 
the true and false branch, respectively.

The methods that are needed to be overridden are:

* **eval_rule**: evaluates the rule, i.e., returns True if the rule's condition is fulfilled.
* **eval_all**: evaluates a list of samples (uses numpy to optimize the decision process).
* **to_string**: prints the node to the string.
* **copy**: returns a new instance as a copy of current object.


It is also conveniente to extend **\_\_init\_\_** method to store some other crucial value for a specific node.

Implemented subclasses are AxisRule and LinearRule representing 
axis parallel and linear split node. There is also implemented Leaf - 
the leaf node of the tree.

It is possible to implement other subclasses of the class Rule however in the tree structure
leafs should be always of the class Leaf because the algorithms implemented in RuleTree class
are dependent on that property.

### class RuleTree

The main class of this subpackage. It represents general decision tree.

This class is not encapsulated. It is rather free to operate on. Its implementation
serves two purpouses. The first is a storage of the general information about the tee such as

- root node,
- number of classes,
- number of input values.

And the second purpose is to implement some simple operation on the tree structure.

1. static methods
    * rt.half_in_ruletree(number_of_classes): creates tree with the same size as the
    number of classes. This tree makes classification based on the first value that
    is grater than 0.5.
    * rt.load(filename): loads RuleTree object from the pickle file.
2. evaluation
    * rt.eval_one(x): evaluates one sample.
    * rt.eval_all(x, all_nodes=None, prevs=None): evaluates all samples stored in the list x.
    When arguments all_nodes and prevs are passed then the method do not need to
    do recursive calls and its faster (it also holds for other methods with these
    arguments).
3. copy, save, and print
    * rt.copy(): makes deep copy of the tree.
    * rt.save(filename): saves ruletree into the pickle file.
    * rt.to_string(): prints ruletree. Be aware of the computation complexity of this
    function when the tree is big.
    * rt.print_expanded_rules(): prints all rules, i.e., each path from 
    the root to a leaf node as If-THEN rule with conjuction of the nodes conditions.
    * rt.view_graph(): returns graphviz Digraph. It also can show it and store it as pdf.
4. getters
    * rt.get_all_nodes(): returns all nodes that decision graph contains starting
    with the rt.root.
    * rt.get_predecessor_dict(): returns a map between a node and a list of their predecessors.
    * rt.get_all_expanded_rules(): returns list of all paths in the graph that starts 
    in the rt.root and ends in a leaf node.
    * rt.get_thresholds(): for the trees with Axis rules returns all thresholds
    for each attribute
    * rt.get_rule_index_dict(x): returns a map between a node and indexes
    of samples x that visited the node during the evaluation.
    * rt.get_stats(): Return some chosen properties of the ruletree:
            number of rules, number of nodes, number of used indexes 
            by the tree (supports only AxisRule and LinearRule), 
            a sum of the lenght of all rules, a sum of indexes used 
            in each rule.
5. graph operations
    * rt.replace_leaf_with_set(leaf_class_set, replacing_rule, all_nodes=None, prevs=None): replaces all
    leafs with the class set equal to leaf_class_set by replacing rule.
    * rt.replace_rule(matching_rule, replacement, prev_entries=None): replaces matching_rule by the replacement.
    If prev_entries are pressent then it do not recursively looking for matching_rule
    in the graph. (Hint: ```prev_entries = prevs\[matching_rule\]```)
    * rt.delete_node(node, branch, all_nodes, prevs): deletes the node in graph structure
    and replace it by its branch specified with the argument.
    * rt.fill_class_sets(): fills class sets accordingly to the leaf class sets.
    Node with two Leafs in branches with class sets {1} and {0,3} lead will have
    class set {0,1,3}.
    * rt.fill_hits(x, labels, remove_redundat=False): fills class_hits in all nodes
    in the graph by evaluating samples x with their true labels. If remove_redundat
    is True then the nodes that are not filled are removed.
6. checks
    * rt.check_none_inside_tree(): returns True if only Leaf nodes has None in some branch.
    * rt.check_one_evaluation(): checks if the graph evalueates each sample to only one class.
    * rt.graph_max_indegree(all_nodes=None, return_all=False): returns maximal indegree
    occurring in the graph. And returns a map between a node and its indegree if return_all is True.
7. condition pruning
    * rt.fill_hits with remove_redundat=True option.
    * rt.delete_interval_redundancy(): Deletes the nodes which result 
    is forced by conditions of the previous nodes in the path from the rt.root to it.
    * rt.delete_class_redundacy(): Deletes the parts of the 
    RuleTree structure which lead to the same result (classification).
    * rt.remove_unvisited(visited_rules): removes other nodes that specified in argument.
    * rt.remove_unvisited_edges(all_nodes, prevs): removes the nodes that has num_true or num_false equal to zero.
    * rt.prune_using_hits(min_split_fraction=0.0, min_rule_samples=1, all_nodes=None):
    Removes all nodes that have sum of class_hits less than min_rule_samples and
    ```max(num_true/num_false, num_false/num_true) < min_split_fraction```.

### Building RuleTree

Standard method to built DT is based on researching all possible splits and taking
that minimizes weighted sum of impurities of its division.
The main concept of this algorithm is implemented in *build* module by function
**build_from_rules**.

Required arguments are
* rule_list: a list of rules, e.g., object with a class AxisRule
* data: a list of inputs of training samples
* labels: a list of labels of training samples

Warning: Do not use this function for more than 100 possible nodes. It is
general and not optimized. If you want to train standard DT use sklearn and then
convert the result by **sklearndt_to_ruletree** function in *ulits* module. 
(Even ANN-DT algorithm can be set to build DT in the standard way and it is optimized)

### utils.py

Main reason for this module is a conversion between other models of DT and the RuleTree class.

**sklearndt_to_ruletree** converts sklearn model into the RuleTree object. It is also possible
to add some additional pruning setting like min_split_fraction (see rt.prune_using_hits).

**pybliquedt_to_ruletree** and **train_OC1** assumes a package [*pyblique*](https://github.com/KDercksen/pyblique).
They produce a RuleTree with LinearRule splits.


### rutils.py

This module required *rpy2* and R-project installed. It provides functions to train C5.0 DT
and convert it to the RuleTree object.

## ANN-DT method (anndt subpackage)

This method is very similar to the standard DT building, but it at each node looks
if there is enough samples and if not new samples are generated by sampling input space
and taking the output of the NN. There are also other impurity functions because the
NN provides continuous output for samples and not just label.

The main function is **anndt(model_fun, x, params, MeasureClass=None, stat_test=None, sampler=None, init_restrictions=None):**,
where arguments are:
* **model_fun** is a function that returns output of the NN for a list of samples when is called.
* **x** a list of training samples.
* **MeasureClass** class of the impurity measure (defined in module *measures*).
* **stat_test** a function of used statistical test (some functions are defined in module *stat_test*).
* **sampler** a subclass of Sampler defined in module *sampling*, where some examples are implemented.
Pay most of the attention to setting or implementing this class for your data.
* **init_restriction** a list of minimal and maximal value of the input.
* **params** a dictionary of additional parameters the main parameters are:
    * **min_samples**: minimal number of samples for creating a new node.
    This values is used to generate that many new samples to fulfill the condition.
    * **min_train**: minimal number of train samples to create a new node. If
    the number of training samples is lower building current subtree stops.
    * **max_depth**: maximal depth of a generated DT.
    * **min_split_fraction**: one of the stopping criteria see rt.prune_using_hits for more details.
    * **varbose**: a level of varbose
        * 0: nothing is printed:
        * 1: some information is printed.
        
        
### module measures.py

In this module the abstract class *MeasureOptimized* is defined. It represent abstraction
that uses ANN-DT for searching for the optimal split. The main method of the class
is *find_split*.

There are also implemented subclasses of *MeasureOptimized*:
* **GiniMeasure**: representing gini index impurity based training algorithm.
* **EntropyMeasure**: representing entropy impurity based training algorithm.
* **FidelityGain**: representing fidelity gain based training algorithm. Fidelity
gain is defined as a percentage of wrongly classified samples by the most occurring
class in each branch.
* **VarianceMeasure**: this impurity can be used **only for a binary classification**.
It computes a variance as the impurity measure.

### module sampling.py

This module defines an abstract class *Sampler* and its two subclasses *NormalSampler* and *BerNormalSampler*.
There are two abstract methods that have to be overwritten:
* get_default_params(self, x): a method that sets up parameters required for sampler.
* generate_x(self, train_x, number, restrictions): a method that generates samples.
For its generation there are available training samples of current node and restriction
defined by input space and previous nodes.

**NormalSampler** defines sampling class for randomly distributed data. It is
convenient to use some statistical test to determine wheater your data are normally
distributed.
 
In **BerNormalSampler** the sampling is conducted by multiplication of Bernoulli and normal distribution.
This results in the values that are either zero or normally distributed.
So, there are three parameters: *p* for the Bernoulli distribution (a probability of
the occurrence of the zero value); *σ* and *μ* for normal distribution.


## DeepRED method (deepred subpackage)

A method that uses activation of all layers of the NN. The algorithm takes 
all activation for training samples. It starts from the last layer. it build DT
that classifies samples base on the last activations. Then for each node
the DT is built that classifies if the node's condition hold or not. After that
the node is substituted by this DT. This process is repeated until the first layer
is reached. Substitution of the nodes by the tree leads to the decision graph.

The main function is **deepred(layers_activations, params)**.
**layers_activations** is a tensor with dimensions (layers x samples x activations).
An argument **params** specifies all setting of the method the main one are:

* **initial_dt_train_params**: a dictionary of the parameters that are used for
training the initial dt by sklearn package.
* **dt_train_params**: a dictionary of the parameters that are used for
training other than initial dt by sklearn package.
* **build_first**: if True then the DT is builded on the last activations else
a tree with splits 0.5 is used (see half_in_ruletree method on RuleTree class).
* **varbose**: a level of varbose
    * 0: notning is shown or printed.
    * 1: the main processing informatin is printed.
    * 2: additional information about subtrees is printed.
    * 3: graph of each step is generated as pdf along with the result. 

Function **deepred** returns additional information about the process along with
the extracted RuleTree.

The tensorflow model of the multilayer perceptron that returns all 
activations is included in model.py package. Its name is DeepRedFCNet.

## HypInv method (hypinv subpackage)

The HypInv deals with decision boundaries defined by the NN (generally classifier).
Decision boundaries are parts of the input space where the classification changes.
The method works as follows. Firstly, the wrongly classified point is found. Then,
the closes point on the decision boundary to that point is found. A new linear
split is make as a perpendicular hyperplane to the line between those points
intersecting with the point on the boundary. The rules are created in the form
of the decision tree with linear splits. Building is done by selection the best
splits at each step in the impurity manner taking only generated linear splits into account.

There are three available methods for finding the closest point on the decision
boundary. These methods are:
1. generating evenly distributed points on the decision boundary by evolutionary algorithm,
2. inversion of the NN followed by sliding along the boundary,
3. modified cost function.

Their usage depends on the dimensionality of the input. The boundaries between 
these three methods is roughly 60, 300. So, in dimension under 60 the generation
point on the boundary can be efficient, but not for dimension 500 where only
recommended method is to use modified cost function.

This method was used only on data with high dimensional spaces so
the evolutionary search for points on the decision boundary is not supported.

The main function is **hypinv(model, x, params=dict(), input_range=None)**.
* **model** an object with subclass NetForHypinv (in module hypinv.model).
It also determine if the modified loss is used.
* **x** a list of training samples (just inputs).
* **input_range** a range of the input space, e.g., [[0, 255] for i in range(dimensions)] for one bite inputs.
If is not defined the algorithm set it by taking maximal and minimal value of the attributes of training samples.
* **params** specifies all setting of the method the main one are:
    * **max_rules** maximal number of linear splits that will be created.
    * **max_depth** maximal depth of the generated tree.
    * **thresh_fidelity** if fidelity exceeds this value then the algorithm stops and return current RuleTree.
    * **use_training_points** if True then search for badly classified points is done in the training set else 
    the algorithm uniformly samples input space to find point where the classification
    of the NN and current RuleTree is different. 
    * **max_sliding_steps** the maximal number of steps in the sliding procedure.
    * **gtrain_params** a dictionary with gtrain parameters that are used for
    optimization modified cost function.

Function **hypinv** returns additional information about the process along with
the extracted RuleTree.


## Future work


I have noticed that there are no recent attempts to extend decision trees (DTs).
Eventhough, the HypInv rule extraction method is using linear splits 
and achieves very good results in cases where other methods or DTs fails. 
So, DT with linear (oblique) splits can be very promising direction
of research. However, general linear splits are hard to explain. Therefore, a new
method should have following properties:

2. Good separation of classes - it is the same properties as for DTs
2. Low number of attributes in combination

The idea is to define parametric function and by its optimization find desired
attributes of the linear combination. However, plain iterative optimization will
do not lead to setting some attributes zeros which is required by second property.


