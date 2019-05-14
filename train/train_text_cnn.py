import tensorflow as tf
import numpy as np
import os
import datetime
import time

import data_helpers

from sklearn.metrics import precision_recall_fscore_support
import warnings
import sklearn.exceptions

from configs.configure import FLAGS
from models.Text_CNN import TextCNN

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def train():
    with tf.device('/cpu:0'):
        x_text, y, pos1, pos2 = data_helpers.load_data_and_labels(FLAGS.train_path)

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length

    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    x = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
    # [95 96 97 98 99 100 101 999 999 999 ... 999]
    # =>
    # [11 12 13 14 15  16  21  17  17  17 ...  17]
    # dimension = MAX_SENTENCE_LENGTH

    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    pos_vocab_processor.fit(pos1 + pos2)
    p1 = np.array(list(pos_vocab_processor.transform(pos1)))
    p2 = np.array(list(pos_vocab_processor.transform(pos2)))
    print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    print("position_1 = {0}".format(p1.shape))
    print("position_2 = {0}".format(p2.shape))
    print("")

    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    p1_shuffled = p1[shuffle_indices]
    p2_shuffled = p2[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    p1_train, p1_dev = p1_shuffled[:dev_sample_index], p1_shuffled[dev_sample_index:]
    p2_train, p2_dev = p2_shuffled[:dev_sample_index], p2_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                text_vocab_size=len(text_vocab_processor.vocabulary_),
                text_embedding_size=FLAGS.text_embedding_dim,
                pos_vocab_size=len(pos_vocab_processor.vocabulary_),
                pos_embedding_size=FLAGS.pos_embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(cnn.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
            pos_vocab_processor.save(os.path.join(out_dir, "pos_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.embedding_path:
                pretrain_W = utils.load_word2vec(FLAGS.embedding_path, FLAGS.text_embedding_dim, text_vocab_processor)
                sess.run(cnn.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, p1_train, p2_train, y_train)),
                                              FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for batch in batches:
                x_batch, p1_batch, p2_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    cnn.input_text: x_batch,
                    cnn.input_p1: p1_batch,
                    cnn.input_p2: p2_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        cnn.input_text: x_dev,
                        cnn.input_p1: p1_dev,
                        cnn.input_p2: p2_dev,
                        cnn.input_y: y_dev,
                        cnn.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    pre, rec, f1, _ = precision_recall_fscore_support(np.argmax(y_dev, axis=1), predictions,
                                                                      labels=np.array(range(1, 19)), average="macro")

                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(f1))
                    print("[UNOFFICIAL] (2*9+1)-Way Macro-Average recall Score (excluding Other): {:g}\n".format(rec))
                    print(
                        "[UNOFFICIAL] (2*9+1)-Way Macro-Average precision Score (excluding Other): {:g}\n".format(pre))

                    # Model checkpoint
                    if best_f1 < f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
