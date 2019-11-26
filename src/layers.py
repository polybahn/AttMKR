import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Layer(object):
    def __init__(self, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.vars = []

    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs

    @abstractmethod
    def _call(self, inputs):
        pass


class Dense(Layer):
    def __init__(self, input_dim, output_dim, dropout=0.0, act=tf.nn.relu, name=None):
        super(Dense, self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        with tf.variable_scope(self.name):
            self.weight = tf.get_variable(name='weight', shape=(input_dim, output_dim), dtype=tf.float32)
            self.bias = tf.get_variable(name='bias', shape=output_dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight]

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        output = tf.matmul(x, self.weight) + self.bias
        return self.act(output)


class AttentionUnit(Layer):
    def __init__(self, dim, channel, use_bias=True, name=None):
        super(AttentionUnit, self).__init__(name)
        self.use_bias = use_bias
        self.dim = dim
        self.c = channel
        with tf.variable_scope(self.name):
            self.weight_f = tf.get_variable(name='weight_f', shape=(1, channel), dtype=tf.float32)
            self.weight_g = tf.get_variable(name='weight_g', shape=(1, channel), dtype=tf.float32)
            self.bias_f = tf.get_variable(name='bias_f', shape=channel, initializer=tf.zeros_initializer())
            self.bias_g = tf.get_variable(name='bias_g', shape=channel, initializer=tf.zeros_initializer())
            self.weight_h = tf.get_variable(name='weight_h', shape=(1, channel), dtype=tf.float32)
            self.weight_l = tf.get_variable(name='weight_l', shape=(1, channel), dtype=tf.float32)
            self.bias_h = tf.get_variable(name='bias_h', shape=channel, initializer=tf.zeros_initializer())
            self.bias_l = tf.get_variable(name='bias_l', shape=channel, initializer=tf.zeros_initializer())
            self.weight_m = tf.get_variable(name='weight_m', shape=(channel, 1), dtype=tf.float32)
            self.weight_n = tf.get_variable(name='weight_n', shape=(channel, 1), dtype=tf.float32)
            self.bias_m = tf.get_variable(name='bias_m', shape=dim, initializer=tf.zeros_initializer())
            self.bias_n = tf.get_variable(name='bias_n', shape=dim, initializer=tf.zeros_initializer())
            self.gamma = tf.get_variable(name="gamma", shape=1, initializer=tf.zeros_initializer())
        self.vars = [self.weight_f, self.weight_g, self.weight_h, self.weight_l, self.weight_m, self.weight_n, self.gamma]

    def _call(self, inputs):
        # [batch_size, dim]
        ori_v, ori_e = inputs

        # [batch_size * dim, 1], [batch_size * dim, 1]
        v = tf.reshape(ori_v, [-1, 1])
        e = tf.reshape(ori_e, [-1, 1])

        # [batch_size, dim, c], [batch_size, dim, c]
        f_v = tf.reshape(tf.matmul(v, self.weight_f), [-1, self.dim, self.c])
        g_e = tf.reshape(tf.matmul(e, self.weight_g), [-1, self.dim, self.c])
        f_v = tf.nn.bias_add(f_v, self.bias_f) if self.use_bias else f_v
        g_e = tf.nn.bias_add(g_e, self.bias_g) if self.use_bias else g_e

        # [batch_size, dim, c], [batch_size, dim, c]
        h_v = tf.reshape(tf.matmul(v, self.weight_h), [-1, self.dim, self.c])
        l_e = tf.reshape(tf.matmul(e, self.weight_l), [-1, self.dim, self.c])
        h_v = tf.nn.bias_add(h_v, self.bias_h) if self.use_bias else h_v
        l_e = tf.nn.bias_add(l_e, self.bias_l) if self.use_bias else l_e

        # [batch_size, dim, dim], [batch_size, dim, dim]
        s_matrix = tf.matmul(g_e, tf.transpose(f_v, perm=[0, 2, 1]))
        s_matrix_transpose = tf.transpose(s_matrix, perm=[0, 2, 1])

        # [batch_size, dim, dim], [batch_size, dim, dim]
        beta_ev = tf.nn.softmax(s_matrix)
        beta_ve = tf.nn.softmax(s_matrix_transpose)

        # [batch_size * dim, c]
        o_v = tf.reshape(tf.matmul(beta_ev, h_v), [-1, self.c])
        o_e = tf.reshape(tf.matmul(beta_ve, l_e), [-1, self.c])

        # [batch_size, dim]
        o_v = tf.reshape(tf.matmul(o_v, self.weight_m), [-1, self.dim])
        o_e = tf.reshape(tf.matmul(o_e, self.weight_n), [-1, self.dim])
        o_v = tf.nn.bias_add(o_v, self.bias_n) if self.use_bias else o_v
        o_e = tf.nn.bias_add(o_e, self.bias_m) if self.use_bias else o_e

        #
        v_output = ori_v + self.gamma * o_v
        e_output = ori_e + self.gamma * o_e

        return v_output, e_output





class CrossCompressUnit(Layer):
    def __init__(self, dim, name=None):
        super(CrossCompressUnit, self).__init__(name)
        self.dim = dim
        with tf.variable_scope(self.name):
            self.weight_vv = tf.get_variable(name='weight_vv', shape=(dim, 1), dtype=tf.float32)
            self.weight_ev = tf.get_variable(name='weight_ev', shape=(dim, 1), dtype=tf.float32)
            self.weight_ve = tf.get_variable(name='weight_ve', shape=(dim, 1), dtype=tf.float32)
            self.weight_ee = tf.get_variable(name='weight_ee', shape=(dim, 1), dtype=tf.float32)
            self.bias_v = tf.get_variable(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable(name='bias_e', shape=dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight_vv, self.weight_ev, self.weight_ve, self.weight_ee]

    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = tf.expand_dims(v, dim=2)
        e = tf.expand_dims(e, dim=1)

        # [batch_size, dim, dim]
        c_matrix = tf.matmul(v, e)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])

        # [batch_size, dim]
        v_output = tf.reshape(tf.matmul(c_matrix, self.weight_vv) + tf.matmul(c_matrix_transpose, self.weight_ev),
                              [-1, self.dim]) + self.bias_v
        e_output = tf.reshape(tf.matmul(c_matrix, self.weight_ve) + tf.matmul(c_matrix_transpose, self.weight_ee),
                              [-1, self.dim]) + self.bias_e

        return v_output, e_output
