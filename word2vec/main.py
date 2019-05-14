import pickle
import re

import pandas as pd
import gensim
import os
from gensim.models import word2vec
from sklearn.manifold import TSNE
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
from tqdm import  tqdm

from word2vec.plot import plt_3d, plt_2d
from word2vec.wordcut import cut_words

dir_corpus = r'C:\Users\acer\Documents\Tencent Files\894371607\FileRecv\环球网\环球网'
model_dir = '../dataset/hqw_news_word2vec.model'
word_freq_path = '../dataset/hqw_word_freq.pkl'
sentences_path = '../dataset/hqw_sentences.pkl'


def train_model(sentences):
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=256)
    model.save(model_dir)
    return model


def tsne_show(word_freq, model, num):
    words_except_quotes = [x[0] for x in word_freq if re.match('\w', x[0])]
    high_freq_words = words_except_quotes[0:num]
    # print(high_freq_words)
    X_tsne = TSNE(n_components=2, learning_rate=100, verbose=1).fit_transform(model.wv[high_freq_words])
    print('draw the figure')
    plt_2d(X_tsne, high_freq_words)


def main():
    if os.path.exists(sentences_path) and os.path.exists(word_freq_path):
        with open(sentences_path, 'rb') as f:
            sentences = pickle.load(f)
        with open(word_freq_path, 'rb') as f:
            word_freq = pickle.load(f)
    else:
        sentences, word_freq = cut_words(dir_corpus=dir_corpus)
    if os.path.exists(model_dir):
        model = word2vec.Word2Vec.load(model_dir)
    else:
        model = train_model(sentences)
    tsne_show(word_freq, model, num=150)


if __name__ == '__main__':
    main()
