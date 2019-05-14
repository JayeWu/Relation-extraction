import tensorflow as tf
import numpy as np
import os
import datetime
import time

from sklearn.metrics import f1_score, precision_recall_fscore_support

from configs.configure_bigru import FLAGS
from models.BIGRU_INCEP import BiGRUINCEP
from models.BiGRU_CNN import BiGRUCNN
from models.att_lstm import AttLSTM

from data_helpers import data_helper

pos_dir = r'D:\rde\enterprise_relation_extraction\positive_labeled\city.txt'
neg_dir = r'D:\rde\enterprise_relation_extraction\negative_labeled\city1.txt'


# pos_dir = r'E:\wjy_projects\enterprise_relation_extraction\positive_labeled\labeled_city.txt'
# neg_dir = r'E:\wjy_projects\enterprise_relation_extraction\negative_labeled\neg_labeled_city.txt'

def train():
    max_sequence_length = 50
    #  读取txt文件，获取训练数据集
    with tf.device('/cpu:0'):
        x_text_posi, y_posi, pos1s_posi, pos2s_posi = data_helper.load_data_and_labels(pos_dir, True)
        x_text_neg, y_neg, pos1s_neg, pos2s_neg = data_helper.load_data_and_labels(neg_dir, False)
        print("positive samples nums /negative samples nums:  {:d}/{:d}\n".format(len(y_posi), len(y_neg)))
        x_text = x_text_posi + x_text_neg
        pos1s = pos1s_posi + pos1s_neg
        pos2s = pos2s_posi + pos2s_neg
        y = np.concatenate((y_posi, y_neg), axis=0)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                         tokenizer_fn=chinese_tokenizer)

    # x为输入句子集，（句子个数，句子长度）
    # y为输入句子的标签集， （句子总数，标签种类（2））
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))

    # Randomly shuffle result to split into train and test(dev)
    np.random.seed(10)
    # 打乱句子顺序
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # 分隔测试数据集和训练数据集，10折交叉验证

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = BiGRUINCEP(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

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
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # # Pre-trained word2vec
            # if FLAGS.embedding_path:
            #     pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
            #     sess.run(model.W_text.assign(pretrain_W))
            #     print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                # print(x_batch[0])
                feed_dict = {
                    model.input_text: x_batch,
                    model.input_y: y_batch,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    model.cnn_dropout_keep_prob: 0.5
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        model.input_text: x_dev,
                        model.input_y: y_dev,
                        model.emb_dropout_keep_prob: 1.0,
                        model.rnn_dropout_keep_prob: 1.0,
                        model.dropout_keep_prob: 1.0,
                        model.cnn_dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    pre, rec, f_score, _ = precision_recall_fscore_support(np.argmax(y_dev, axis=1), predictions,
                                                                           labels=np.array(range(1, 2)),
                                                                           average='macro')

                    f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, 2)), average='macro')
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                    print("Precision Score (excluding Other): {:g}\n".format(pre))
                    print("Recall Score Score (excluding Other): {:g}\n".format(rec))
                    print("F1 Score (excluding Other): {:g}\n".format(f_score))
                    # print("F1 Score (excluding Other): {:g}\n".format(f1))

                    # Model checkpoint
                    if best_f1 < f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


def chinese_tokenizer(docs):
    for doc in docs:
        yield list(doc)


if __name__ == '__main__':
    train()
