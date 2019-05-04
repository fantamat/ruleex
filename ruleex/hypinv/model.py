from gtrain import Model
import numpy as np
import tensorflow as tf


class NetForHypinv(Model):
    """
    Implementaion of the crutial function for the HypINV algorithm.
    Warning: Do not use this class but implement its subclass, for example see FCNetForHypinv
    """
    def __init__(self, weights):
        self.eval_session = None
        self.grad_session = None
        self.initial_x = None
        self.center = None
        self.weights = weights
        self.out_for_eval = None #(going to be filled in build_for_eval method)
        self.boundary_out_for_eval = None
        self.trained_x = None
        self.training_class_index = None
        self.x = None # tf variable for inversion (going to be filled in build method)
        self.x_for_eval = None
        self.out = None
        self.boundary_out = None # list of tf tensorf for each class of softmax class vs others output
        self.loss = None
        self.boundary_loss = None
        self.t = None #target
        self.boundary_t = None
        self.x1 = None # this attribute is used of purposes of modified loss function


    def __del__(self):
        # close arr sessions
        if self.eval_session:
            self.eval_session.close()
        if self.grad_session:
            self.grad_session.close()

    def set_initial_x(self, initial_x):
        # sets starting point for the search of the closest point
        self.initial_x = initial_x

    def set_center(self, center):
        # sets center point
        self.center = center / np.linalg.norm(center)

    def set_x1(self, x1):
        # sets x1 to which we want to found the cosest point x0
        self.x1 = x1

    def has_modified_loss(self):
        pass # if uses modified loss then it returns true

    def set_initial_x_in_session(self, x, session=None):
        # sets initial x in certain session
        if session is None:
            self.set_initial_x(x)
        else:
            pass # overide this method

    def eval(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1,len(x)))
        if not self.eval_session:
            self.eval_session = tf.Session()
            with self.eval_session.as_default():
                self.build_for_eval()
            self.eval_session.run(tf.global_variables_initializer())
        return self.eval_session.run(self.out_for_eval, {self.x_for_eval: x})

    def boundary_eval(self, x, class_index):
        # evaluates binary classificaitons class_index and other classes
        if not self.eval_session:
            self.eval_session = tf.Session()
            with self.eval_session.as_default():
                self.build_for_eval()
            self.eval_session.run(tf.global_variables_initializer())
        return self.eval_session.run(self.boundary_out_for_eval[class_index], {self.x_for_eval: x})

    def get_boundary_gradient(self, x, class_index):
        # computes gradient of the boundary for specified class_index
        if not self.grad_session:
            self.grad_session = tf.Session()
            with self.grad_session.as_default():
                self.build_for_eval()
                self.grad = list()
                for i in range(len(self.weights[0][-1][0])):
                    self.grad.append(tf.gradients(self.boundary_out_for_eval[i], [self.x_for_eval])[0])
                self.grad_x = self.x_for_eval
        return self.grad_session.run(self.grad[class_index], {self.grad_x: x})

    def build_for_eval(self):
        # build model for evaluation
        pass #override this method (fill self.out_for_eval)

    def train_ended(self, session):
        self.trained_x = session.run(self.x)

    def build(self):
        # build model for training
        pass #override this method (fill self.x, self.out)

    def set_train_class(self, class_index):
        # sets class of the x1
        self.training_class_index = class_index

    # overided methods from gtrain.Model
    def get_loss(self):
        if self.training_class_index is None:
            return self.loss
        else:
            return self.boundary_loss[self.training_class_index]

    def get_hits(self):
        return self.get_loss()

    def get_count(self):
        return self.get_loss()

    def get_train_summaries(self):
        return []

    def get_dev_summaries(self):
        return []

    def get_placeholders(self):
        if self.training_class_index is None:
            return [self.t]
        else:
            return [self.boundary_t]



#________________________________________EXAMPLES_OF_NetForHypinv_CLASS_____________________________________________

class FCNetForHypinv(NetForHypinv):
    """
    Implementation of multi layer perceptron to by used in HypINV rule extraction algorithm
    """
    def __init__(self, weights, function=tf.sigmoid, use_modified_loss=False, mu = 0.01):
        """
        :param weights: saved as [list of weights for layers][0 weight, 1 bias]
        :param function: tf function for propagation. For example tf.nn.sigmoid, tf.atan
        :param use_modified_loss: weather the modified loss should be used
        :param mu: factor of the penalty terms that specified the distance between x0 and x1 and
            the distance x1 from the boundary
        """
        super(FCNetForHypinv, self).__init__(weights)
        self.function = function
        self.layer_sizes = [len(self.weights[0][0])]
        for bias in weights[1]:
            self.layer_sizes.append(len(bias))
        self.num_classes = self.layer_sizes[-1]
        self.initial_x = np.zeros([1, self.layer_sizes[0]])
        self.use_modified_loss = use_modified_loss
        self.mu = mu

    def build(self):
        with tf.name_scope("Input"):
            if self.center is not None:
                self.point_weights = tf.Variable(self.center.reshape((1, len(self.center))),
                                                 dtype=tf.float64, trainable=False, name="Boundary_point")
                init_factor = self.center
                init_factor[init_factor!=0] = self.initial_x[init_factor!=0] / self.center[init_factor!=0]
                self.factor = tf.Variable(init_factor.reshape((1, len(self.center))),
                                         dtype=tf.float64, name="factor")
            else:
                self.point_weights = tf.Variable(self.initial_x.reshape((1, len(self.initial_x))),
                                                 dtype=tf.float64, trainable=False, name="Boundary_point")
                self.factor = tf.Variable(np.ones((1, len(self.center))),
                                          dtype=tf.float64, name="factor")
            self.x = self.point_weights * self.factor
        with tf.name_scope("Target"):
            if self.use_modified_loss:
                x1_constant = tf.constant(self.x1.reshape((1, len(self.x1))), dtype=tf.float64)
            self.t = tf.placeholder(tf.float64, shape=[None, self.num_classes], name="Target_output")
            self.boundary_t = tf.placeholder(tf.float64, shape=[None, 2], name="Target_boundary_output")
        with tf.name_scope("FC_net"):
            flowing_x = self.x
            for i, _ in enumerate(self.weights[0]):
                with tf.name_scope("layer_{}".format(i)):
                    W = tf.constant(self.weights[0][i], name="Weight_{}".format(i), dtype=tf.float64)
                    b = tf.constant(self.weights[1][i], name="Bias_{}".format(i), dtype=tf.float64)
                    flowing_x = self.function(tf.nn.xw_plus_b(flowing_x, W, b))
            y = flowing_x
            self.out = tf.nn.softmax(y)
        with tf.name_scope("Binary_class_output"):
            self.boundary_out = list()
            for i in range(self.num_classes):
                mask = True+np.zeros(self.num_classes, dtype=np.bool)
                mask[i] = False
                x0 = self.out[:,i]
                x1 = tf.reduce_max(tf.boolean_mask(self.out, mask, axis=1), axis=1)
                s = x0+x1
                out = tf.stack([x0/s, x1/s], axis=1)
                self.boundary_out.append(out)
        with tf.name_scope("Loss_functions"):
            self.loss = tf.reduce_mean(
                tf.nn.l2_loss(self.out-self.t),
                name="loss")
        with tf.name_scope("Binary_class_loss"):
            self.boundary_loss = list()
            if self.use_modified_loss:
                for i in range(self.num_classes):
                    self.boundary_loss.append(
                        tf.reduce_mean(tf.nn.l2_loss(self.boundary_out[i]-self.boundary_t)) +
                        self.mu * tf.reduce_mean(tf.nn.l2_loss(self.x - x1_constant))
                    )
            else:
                for i in range(self.num_classes):
                    self.boundary_loss.append(
                        tf.reduce_mean(tf.nn.l2_loss(self.boundary_out[i] - self.boundary_t))
                    )

    def set_initial_x_in_session(self, x, session=None):
        if session is None:
            self.set_initial_x(x)
        else:
            if self.center is None:
                session.run([
                    self.point_weights.assign(x.reshape((1, len(x)))),
                    self.factor.assign(np.ones((1, len(x))))
                ])
            else:
                init_factor = self.center
                init_factor[init_factor!=0] = x[init_factor!=0] / self.center[init_factor!=0]
                session.run(self.factor.assign(init_factor.reshape((1,len(init_factor)))))

    def build_for_eval(self):
        with tf.name_scope("eInput"):
            self.x_for_eval = tf.placeholder(tf.float32, shape=[None, len(self.weights[0][0])])#tf.Variable(tf.constant(self.initial_x), name="Boundary_point")
        with tf.name_scope("eFC_net"):
            flowing_x = self.x_for_eval
            for i, _ in enumerate(self.weights[0]):
                W = tf.constant(self.weights[0][i], name="eWeight_{}".format(i))
                b = tf.constant(self.weights[1][i], name="eBias_{}".format(i))
                flowing_x = self.function(tf.nn.xw_plus_b(flowing_x, W, b), name="elayer_{}".format(i))
            y = flowing_x
            self.out_for_eval = tf.nn.softmax(y)
        with tf.name_scope("Binary_class_output"):
            self.boundary_out_for_eval = list()
            for i in range(self.num_classes):
                mask = True+np.zeros(self.num_classes, dtype=np.bool)
                mask[i] = False
                x0 = self.out_for_eval[:, i]
                x1 = tf.reduce_max(tf.boolean_mask(self.out_for_eval, mask, axis=1), axis=1)
                s = x0+x1
                out = tf.stack([x0/s, x1/s], axis=1)
                self.boundary_out_for_eval.append(out)

    def has_modified_loss(self):
        return self.use_modified_loss

    def name(self):
        return "Hypinv_FC_net_{}".format("-".join([str(ls) for ls in self.layer_sizes]))



class FCNetForHypinvBinary(FCNetForHypinv):
    """
    Implementation of multi layer perceptron to by used in HypINV rule extraction algorithm
    The task is simplified to the binary classificaiton base_class_index against the other classes
    """
    def __init__(self, weights, base_class_index, function=tf.sigmoid, use_modified_loss=False, mu = 0.01):
        """
        :param weights: saved as [list of weights for layers][0 weight, 1 bias]
        :param base_class_index: an index of the class which is used as the base class
        :param function: tf function for propagation. For example tf.nn.sigmoid, tf.atan
        :param use_modified_loss: weather the modified loss should be used
        :param mu: factor of the penalty terms that specified the distance between x0 and x1 and
            the distance x1 from the boundary
        """
        super(FCNetForHypinvBinary, self).__init__(weights)
        self.base_class_index = base_class_index
        self.function = function
        self.layer_sizes = [len(self.weights[0][0])]
        for bias in weights[1]:
            self.layer_sizes.append(len(bias))
        self.num_classes = self.layer_sizes[-1]
        self.initial_x = np.zeros([1, self.layer_sizes[0]])
        self.use_modified_loss = use_modified_loss
        self.mu = mu


    def build(self):
        with tf.name_scope("Input"):
            self.init_point = tf.Variable(self.initial_x.reshape((1, len(self.initial_x))),
                                     dtype=tf.float64, trainable=False, name="Boundary_point")
            self.factor = tf.Variable(np.ones((1, len(self.initial_x))),
                                     dtype=tf.float64, name="factor")
            self.x = self.init_point * self.factor
        with tf.name_scope("Target"):
            if self.use_modified_loss:
                x1_constant = tf.constant(self.x1.reshape((1, len(self.x1))), dtype=tf.float64)
            self.t = tf.placeholder(tf.float64, shape=[None, 2], name="Target_output")
            self.boundary_t = tf.placeholder(tf.float64, shape=[None, 2], name="Target_boundary_output")
        with tf.name_scope("FC_net"):
            flowing_x = self.x
            for i, _ in enumerate(self.weights[0]):
                with tf.name_scope("layer_{}".format(i)):
                    W = tf.constant(self.weights[0][i], name="Weight_{}".format(i), dtype=tf.float64)
                    b = tf.constant(self.weights[1][i], name="Bias_{}".format(i), dtype=tf.float64)
                    flowing_x = self.function(tf.nn.xw_plus_b(flowing_x, W, b))
            y = flowing_x
            full_out = tf.nn.softmax(y)
        with tf.name_scope("Binary_class_output"):
            self.boundary_out = list()
            mask = True+np.zeros(self.num_classes, dtype=np.bool)
            mask[self.base_class_index] = False
            x0 = full_out[:,self.base_class_index]
            x1 = tf.reduce_max(tf.boolean_mask(full_out, mask, axis=1), axis=1)
            s = x0+x1
            self.out = tf.stack([x0/s, x1/s], axis=1)
            self.boundary_out.append(self.out)
            self.boundary_out.append(tf.stack([x1/s, x0/s], axis=1))
        with tf.name_scope("Loss_functions"):
            self.loss = tf.reduce_mean(
                tf.nn.l2_loss(self.out-self.t),
                name="loss")
        with tf.name_scope("Binary_class_loss"):
            self.boundary_loss = list()
            if self.use_modified_loss:
                for i in range(2):
                    self.boundary_loss.append(
                        tf.reduce_mean(tf.nn.l2_loss(self.boundary_out[i]-self.boundary_t)) +
                        self.mu * tf.reduce_mean(tf.nn.l2_loss(self.x - x1_constant))
                    )
            else:
                for i in range(2):
                    self.boundary_loss.append(
                        tf.reduce_mean(tf.nn.l2_loss(self.boundary_out[i] - self.boundary_t))
                    )

    def build_for_eval(self):
        with tf.name_scope("eInput"):
            self.x_for_eval = tf.placeholder(tf.float32, shape=[None, len(self.weights[0][0])])#tf.Variable(tf.constant(self.initial_x), name="Boundary_point")
        with tf.name_scope("eFC_net"):
            flowing_x = self.x_for_eval
            for i, _ in enumerate(self.weights[0]):
                W = tf.constant(self.weights[0][i], name="eWeight_{}".format(i))
                b = tf.constant(self.weights[1][i], name="eBias_{}".format(i))
                flowing_x = self.function(tf.nn.xw_plus_b(flowing_x, W, b), name="elayer_{}".format(i))
            y = flowing_x
            full_out = tf.nn.softmax(y)
        with tf.name_scope("Binary_class_output"):
            self.boundary_out_for_eval = list()
            mask = True+np.zeros(self.num_classes, dtype=np.bool)
            mask[self.base_class_index] = False
            x0 = full_out[:, self.base_class_index]
            x1 = tf.reduce_max(tf.boolean_mask(full_out, mask, axis=1), axis=1)
            s = x0+x1
            self.out_for_eval = tf.stack([x0/s, x1/s], axis=1)
            self.boundary_out_for_eval.append(self.out_for_eval)
            self.boundary_out_for_eval.append(tf.stack([x1/s, x0/s], axis=1))

    def get_boundary_gradient(self, x, class_index):
        if not self.grad_session:
            self.grad_session = tf.Session()
            with self.grad_session.as_default():
                self.build_for_eval()
                self.grad = list()
                for i in range(2):
                    self.grad.append(tf.gradients(self.boundary_out_for_eval[i], [self.x_for_eval])[0])
                self.grad_x = self.x_for_eval
        return self.grad_session.run(self.grad[class_index], {self.grad_x: x})

    def has_modified_loss(self):
        return self.use_modified_loss

    def name(self):
        return "Hypinv_FC_net_{}".format("-".join([str(ls) for ls in self.layer_sizes]))


