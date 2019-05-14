import numpy as np
import pandas as pd
import re

pos_labeled_dir = r'E:\wjy_projects\enterprise_relation_extraction\positive_labeled'
neg_labeled_dir = r'E:\wjy_projects\enterprise_relation_extraction\negative_labeled'


def load_data_and_labels(path, is_positive):
    data = []
    pos1s = []
    pos2s = []
    lines = [line.strip() for line in open(path, encoding='utf8')]
    if is_positive:
        rela = 1
    else:
        rela = 0
    for idx in range(0, len(lines)):
        line = lines[idx]
        pp = re.findall('(?<=\<e1\>)[^<]*(?=\</e1\>)', line)
        qq = re.findall('(?<=\<e2\>)[^<]*(?=\</e2\>)', line)
        if (not pp) or (not qq):
            continue
        e1 = pp[0]
        e2 = qq[0]

        # sentence = line.replace('<e1>', ' _e11_ ')
        # sentence = sentence.replace('</e1>', ' _e12_ ')
        # sentence = sentence.replace('<e2>', ' _e21_ ')
        # sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = re.sub('\<e1\>[^<]*\</e1\>', ' _e1_ ', line)
        sentence = re.sub('\<e2\>[^<]*\</e2\>', ' _e2_ ', sentence)
        # sentence = re.sub('\<e1\>[^<]*\</e1\>', '', line)
        # sentence = re.sub('\<e2\>[^<]*\</e2\>', '', sentence)
        if idx % 1000 == 0:
            print(sentence)
        pos1 = sentence.index('_e1_')
        pos2 = sentence.index('_e2_')
        # pos1 = 1
        # pos2 = 1
        pos1s.append(pos1)
        pos2s.append(pos2)
        data.append([idx, sentence, e1, e2, pos1, pos2, rela])
    # print(result)
    # print(len(lines))
    df = pd.DataFrame(data=data, columns=["id", "sentence", "e1", "e2", "pos1", "pos2", "label"])

    # Text Data
    x_text = df['sentence'].tolist()
    # print(x_text)
    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = 2

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

    return x_text, labels, pos1s, pos2s


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("Data Size: {:d}".format(data_size))
    print("num_batches_per_epoch = {0}".format(num_batches_per_epoch))
    print("")
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
