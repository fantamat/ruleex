from gtrain import FCNet
import numpy as np
import tensorflow as tf

from gtrain.model import TextCNN


class DeepRedFCNet(FCNet):
    """
    Model of the fully connected net with its evaluation.
    The binary sub-domain output is also supported by function eval_binary_class.
    The initialization of the weights is done by finishing the training process by gtrain or by call init_eval_weights.
    """

    def init_eval_weights(self, weights):
        self.eval_session = None
        self.weights = weights

    def __del__(self):
        if self.eval_session:
            self.eval_session.close()


    def __eval(self, tensor_str, x):
        x = np.float32(x)
        if not self.eval_session:
            self.eval_session = tf.Session()
            with self.eval_session.as_default():
                self.build_for_eval()
            self.eval_session.run(tf.global_variables_initializer())
        return self.eval_session.run(eval(tensor_str), {self.x_for_eval: x})

    def eval(self, x):
        return self.__eval("self.out_for_eval", x)

    def eval_layers(self, x):
        return self.__eval("self.layers", x)

    def eval_binary_class(self, x, class_index):
        """
        evaluate network that have two dimensional softmax output computed from original specified class output against
        highest output of the other classes
        :param x:
        :param class_index:
        :return:
        """
        return self.__eval("self.out_for_class_eval[{}]".format(class_index), x)

    def build_for_eval(self):
        with tf.name_scope("Input"):
            self.x_for_eval = tf.placeholder(tf.float32, shape=[None, self.input_size], name="Input...")
        with tf.name_scope("FC_net"):
            flowing_x = self.x_for_eval
            self.layers = [flowing_x]
            for i in range(len(self.weights[0])):
                with tf.name_scope("layer_{}".format(i)):
                    W = tf.constant(self.weights[0][i], name="Weights_{}".format(i))
                    b = tf.constant(self.weights[1][i], name="Biases_{}".format(i))
                    flowing_x = self.activation_function(tf.nn.xw_plus_b(flowing_x, W, b))
                    self.layers.append(flowing_x)
            y = flowing_x

            with tf.name_scope("Output"):
                self.out_for_eval = tf.nn.softmax(y)
                self.layers.append(self.out_for_eval)

            with tf.name_scope("Binary_class_output"):
                self.out_for_class_eval = list()
                for i in range(self.layer_sizes[-1]):
                    mask = True+np.zeros(self.layer_sizes[-1], dtype=np.bool)
                    mask[i] = False
                    out = tf.nn.softmax(tf.stack([
                        self.out_for_eval[:,i],
                        tf.reduce_max(
                            tf.boolean_mask(self.out_for_eval, mask, axis=1), axis=1)
                    ], axis=1))
                    self.out_for_class_eval.append(out)

    def train_ended(self, session):
        super().train_ended(session)
        self.init_eval_weights(weights=[self.trained_W, self.trained_b])

    def name(self):
        return "FC_net_for_deepred_{}".format("-".join([str(ls) for ls in self.layer_sizes]))


class DeepRedTextCNN(TextCNN):
    def init_eval_weights(self, weights):
        self.eval_session = None
        self.weights = weights

    def __del__(self):
        if self.eval_session:
            self.eval_session.close()

    def __eval(self, tensor_str, x):
        x = np.float32(x)
        if not self.eval_session:
            self.eval_session = tf.Session()
            with self.eval_session.as_default():
                self.build_for_eval()
            self.eval_session.run(tf.global_variables_initializer())
        return self.eval_session.run(eval(tensor_str), {self.x_for_eval: x})

    def eval(self, x):
        return self.__eval("self.out_for_eval", x)

    def eval_layers(self, x):
        return self.__eval("self.layers", x)

    def eval_binary_class(self, x, class_index):
        """
        evaluate network that have two dimensional softmax output computed from original specified class output against
        highest output of the other classes
        :param x:
        :param class_index:
        :return:
        """
        return self.__eval("self.out_for_class_eval[{}]".format(class_index), x)

    def build_for_eval(self):
        with tf.name_scope("Input"):
            self.tf_emb_for_eval = tf.constant(self.embedding, name="Embedding", dtype=tf.float32)
            self.x_for_eval = tf.placeholder(tf.int32, shape=[None, None], name="Index_input")
        with tf.name_scope("CNN_for_text"):
            filter = tf.constant(self.weights[0][0], name="Filter")
            flowing_x = tf.nn.embedding_lookup(self.tf_emb_for_eval, self.x_for_eval, name="Embedding_layer")
            self.layers = [flowing_x]
            flowing_x = tf.nn.conv1d(flowing_x, filter, 1, "SAME", name="Conv_layer")
            flowing_x = tf.nn.relu(flowing_x)
            self.layers.append(flowing_x)
            flowing_x = tf.reduce_max(flowing_x, axis=1)
            self.layers.append(flowing_x)
            for i in range(len(self.weights[1])):
                with tf.name_scope("layer_{}".format(i)):
                    W = tf.constant(self.weights[0][i+1], name="Weights_{}".format(i))
                    b = tf.constant(self.weights[1][i], name="Biases_{}".format(i))
                    flowing_x = self.activation_function(tf.nn.xw_plus_b(flowing_x, W, b))
                    self.layers.append(flowing_x)
            y = flowing_x

            with tf.name_scope("Output"):
                self.out_for_eval = tf.nn.softmax(y)
                self.layers.append(flowing_x)

            with tf.name_scope("Binary_class_output"):
                self.out_for_class_eval = list()
                for i in range(self.layer_sizes[-1]):
                    mask = True+np.zeros(self.layer_sizes[-1], dtype=np.bool)
                    mask[i] = False
                    out = tf.nn.softmax(tf.stack([
                        self.out_for_eval[:,i],
                        tf.reduce_max(
                            tf.boolean_mask(self.out_for_eval, mask, axis=1), axis=1)
                    ], axis=1))
                    self.out_for_class_eval.append(out)


    def train_ended(self, session):
        super().train_ended(session)
        self.init_eval_weights(weights=[self.trained_W, self.trained_b])

    def name(self):
        return "TextCNN_for_deepred_{}".format("-".join([str(ls) for ls in self.layer_sizes]))

