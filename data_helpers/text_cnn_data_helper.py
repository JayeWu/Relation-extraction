# -*- encoding: utf-8 -*-
# @Software: PyCharm
# @File    : text_cnn_data_helper.py
# @Time    : 2019/4/25 19:32
# @Author  : LU Tianle

"""
"""

import numpy as np
import pandas as pd
import re

type_id_dic = {'100': 1, '101': 2, '102': 3, '103': 4, '104': 5, '106': 6, '107': 7, '108': 8, '109': 9,
               '110': 10, '112': 11, '113': 12, '114': 13, '115': 14, '116': 0}


def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path, encoding='utf8')]
    max_sentence_length = 0
    #  从txt读取文本
    for idx in range(0, 300000):
        infos = lines[idx].split("_!_")
        id = infos[0]
        relation = infos[1]
        sentence = infos[3]
        max_sentence_length = max_sentence_length if max_sentence_length > len(sentence) else len(sentence)
        data.append([id, sentence, relation])
    # print(result)
    # print(len(lines))
    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [type_id_dic[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()
    print(x_text)
    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels, max_sentence_length


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the result at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_relative_position(df, max_sentence_length):
    # Position result
    pos1 = []
    pos2 = []
    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        tokens = nltk.word_tokenize(sentence)
        e1 = df.iloc[df_idx]['e1']
        e2 = df.iloc[df_idx]['e2']

        p1 = ""
        p2 = ""
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            p2 += str((max_sentence_length - 1) + word_idx - e2) + " "
        pos1.append(p1)
        pos2.append(p2)

    return pos1, pos2


if __name__ == "__main__":
    load_data_and_labels('../result/toutiao_cat_data.txt')
