import tensorflow as tf


class BiGRUINCEP:
    """BiGRU-Inception model for relationship extraction or text classification
    author Wu Jinyu 吴锦钰
    bidirectional Gated Recurrent Unit cell and Convolution networks in series

    Args:
      sequence_length: int, the length of input sentence.
      num_classes: int, the number of relations or class.
      vocab_size: int, the size of whole input vocabulary.
      embedding_size: int, the size of word embedding.
      hidden_size: int, The number of units in the GRU cell.
      l2_reg_lambda: float, l2 regulation lambda.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, l2_reg_lambda=0.0, incep_filters=10, entity_vector_length=100):
        # Placeholders for input, output and dropout
        self.input_text_chars = tf.placeholder(tf.string, name='input_text_chars')
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')

        self.entity1 = tf.placeholder(tf.int32, shape=[None, entity_vector_length], name='entity1')
        self.entity2 = tf.placeholder(tf.int32, shape=[None, entity_vector_length], name='entity2')

        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.cnn_dropout_keep_prob = tf.placeholder(tf.float32, name='cnn_dropout_keep_prob')
        initializer = tf.keras.initializers.glorot_normal

        # start ========================== 特征选择及嵌入层  feature select and embedding layer==========================

        # Word Embedding Layer   词嵌入层
        with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)
        # ================================ BiGRU layer===============================
        # Bidirectional GRU
        with tf.variable_scope("bi-gru"):
            _fw_cell = tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                  cell_bw=bw_cell,
                                                                  inputs=self.embedded_chars,
                                                                  sequence_length=self._length(self.input_text),
                                                                  dtype=tf.float32)
            # rnn_outputs, the output of biGRU
            # as a Tensor of shape()
            # 两个gru输出直接加起来
            self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])

        # ================================ Attention layer=============================
        # Attention
        with tf.variable_scope('attention'):
            self.attn, self.alphas = attention(self.rnn_outputs)

        # Dropout
        with tf.variable_scope('dropout_bigru'):
            self.h_drop1 = tf.nn.dropout(self.attn, self.dropout_keep_prob)

        # ================================ Inception layer =============================
        #  inception layer as a parallel layer to bigru layer, constructed with 4 branchs and 3 different cnn units.
        with tf.variable_scope('inception'):
            incep_in = tf.expand_dims(self.embedded_chars, -1)
            # branch1
            self.incep_b1 = tf.layers.conv2d(incep_in, incep_filters, [1, self.rnn_outputs.shape[2]],
                                             kernel_initializer=initializer())
            # branch2
            self.incep_b2_1 = tf.layers.conv2d(incep_in, incep_filters, [1, self.rnn_outputs.shape[2]],
                                               kernel_initializer=initializer())
            self.incep_b2 = tf.layers.conv2d(self.incep_b2_1, incep_filters, [3, 1], kernel_initializer=initializer())

            #  branch3
            self.incep_b3_1 = tf.layers.conv2d(incep_in, incep_filters, [1, self.rnn_outputs.shape[2]],
                                               kernel_initializer=initializer())
            self.incep_b3 = tf.layers.conv2d(self.incep_b3_1, incep_filters, [5, 1], kernel_initializer=initializer())

            #  branch4
            self.incep_b4_1 = tf.layers.max_pooling2d(incep_in, [3, self.rnn_outputs.shape[2]], [1, 1])
            self.incep_b4 = tf.layers.conv2d(self.incep_b4_1, incep_filters, [1, 1], kernel_initializer=initializer())

            self.inceptionOut = tf.concat([tf.layers.flatten(self.incep_b1), tf.layers.flatten(self.incep_b2),
                                           tf.layers.flatten(self.incep_b3), tf.layers.flatten(self.incep_b4)], 1)

        with tf.variable_scope('dropout_incep'):
            self.h_drop2 = tf.nn.dropout(self.inceptionOut, self.dropout_keep_prob)
        # ================================ Fully connected layer =============================
        # Fully connected layer, concat the bigru outputs and inception outputs and entity features vectors
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(tf.concat([self.h_drop1, self.h_drop2], 1), num_classes,
                                          kernel_initializer=initializer())
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2
            # self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    # Length of the sequence result
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length


def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())
    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape

    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas
