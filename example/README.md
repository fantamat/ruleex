# Examples

* Examples of usage of the methods are available in https://drive.google.com/open?id=18FYoK2hdoq6-3WQOtGzsuzPD2dRBhmuV, 
where is the code of experiments of my master thesis along with some preprocessed datasets.

* To run the experiment focus on files `__exp.py` and `__exp_cv.py` (cv means crossvalidation).
Which parameters has to be passes to those scripts is written in comments in the files.

* File `TaT_gtrain.py` contains calls of the ruleex methods.

* Examples tested 2020-05-19 - additional installation of unidecode and graphviz required, also if you encounter error `ValueError: Object arrays cannot be loaded when allow_pickle=False` reinstall numpy to version 1.16.1.

# FAQ

## deepred
with parameters (layers_activations, params)

### Question 1
* *Function description says "layers_activations: a list of numpy arrays",
so it means that I pass in x and y like [x, y]?*

* No, deepred uses all of the activation of the neurons in NN. So, it is necessary to give the method all of it. I suppose that x and y are your inputs and y maybe outputs of the NN. However, the methods need all of the activations, not just the input and the output layer.

* For this reason, I implemented class DeepRedFCNet in rulex.deepred.model, or especially its method eval_layers. You can initialize a object of this class just like your trained NN and then gave it trained weights by method init_eval_weights 
(`nn_deepredfcnet.init_eval_weights([nn_fcnet.trained_W, nn_fcnet.trained_b])`)
Or you can either train the DeepRedFCNet by gtrain from the very beginning.


## anndt
with parameters (model_fun, x, params, MeasureClass=None, stat_test=None, sampler=None, init_restrictions=None)

### Question 1

* *What shell I pass as model_fun parameter?*

*   model_fun is a model function which is from my point of view a callable function that gave you the output for a given input. As I looked into the code, for this purpose, it is also convenient to use DeepRedFCNet because it has implemented function eval. So, your model_fun can be defined as follows: `model_fun = lambda x: nn_deepredfcnet.eval(x)`.

## hypinv
with parameters (model, x, params=dict(), input_range=None)

### Question 1

* *Why I always get the error "TypeError("model must be of the NetForHypinv type!")" in the process of running hypinv.*

* The Hypinv method uses derivative of NN and at some point, it freezes all of the NN weights. So, it needs to have a class that is prepared for this purpose. The abstraction of such prepared class is defined by NetForHypinv (abstract class). 
There is also prepared non-abstract class in ruleex.hypinv.model named FCNetForHypinv. It also requires to pass weights to constructor like weights=[trained_W, trained_b].
