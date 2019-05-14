import tensorflow as tf


class BiGRUCNN:
    """BiGRU-CNN model for relationship extraction or text classification
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
                 hidden_size, l2_reg_lambda=0.0, cnn_kernel_sizes=(1, 3, 5), cnn_filter_nums=(10, 20, 30)):
        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.cnn_dropout_keep_prob = tf.placeholder(tf.float32, name='cnn_dropout_keep_prob')
        initializer = tf.keras.initializers.glorot_normal

        # Word Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
            # 词嵌入层
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)

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
            self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])

        #  multilayrs Cnn
        with tf.variable_scope('multilayrs-cnn'):
            #  cnn layer1
            kernel_size1 = [cnn_kernel_sizes[0], self.rnn_outputs.shape[2]]
            cnn_in1 = tf.expand_dims(self.rnn_outputs, -1)
            self.cnn_out1 = tf.layers.conv2d(cnn_in1, cnn_filter_nums[0], kernel_size1, kernel_initializer=initializer())
            self.pooled1 = tf.nn.max_pool(self.cnn_out1, ksize=[1, 4, 1, 1],
                                          strides=[1, 1, 1, 1], padding='VALID', name="pool")
            print(self.pooled1.shape)

            #  cnn layer2

            kernel_size = [cnn_kernel_sizes[1], 1]
            cnn_in2 = self.pooled1
            self.cnn_out2 = tf.layers.conv2d(cnn_in2, cnn_filter_nums[1], kernel_size, kernel_initializer=initializer())
            self.pooled2 = tf.nn.max_pool(self.cnn_out2, ksize=[1, 4, 1, 1],
                                          strides=[1, 1, 1, 1], padding='VALID', name="pool")
            print(self.pooled2.shape)

            #  cnn layer3

            kernel_size = [cnn_kernel_sizes[2], 1]
            cnn_in3 = self.pooled2
            self.cnn_out3 = tf.layers.conv2d(cnn_in3, cnn_filter_nums[2], kernel_size, kernel_initializer=initializer())
            self.pooled3 = tf.nn.max_pool(self.cnn_out3, ksize=[1, 4, 1, 1],
                                          strides=[1, 1, 1, 1], padding='VALID', name="pool")
            print(self.pooled3.shape)

            self.h_pool_flat = tf.layers.flatten(self.pooled3)
            print(self.h_pool_flat.shape)
        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Fully connected layer
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
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
