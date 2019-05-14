import pickle

import jieba
import os
from tqdm import tqdm

from gensim.models import word2vec

dir_ = r'D:\rde\data\wangyi\公司新闻'
sentences_path = '../dataset/sentences.pkl'
word_freq_path = '../dataset/word_freq.pkl'


def update(word_freq, sentences, cuts_text):
    words = []
    for word in cuts_text:
        words.append(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    sentences.append(words)


def cut_words(dir_corpus=dir_):
    sentences = []
    word_freq = {}

    for path in tqdm(os.listdir(dir_corpus)):
        path = os.path.join(dir_corpus, path)
        if not os.path.isfile(path):
            for file in os.listdir(path):
                file = os.path.join(path, file)
                if os.path.isfile(file):
                    f = open(file, 'r', encoding='utf8')
                    text = f.read().strip()
                    f.close()
                    cuts_text = jieba.cut(text)
                    update(word_freq, sentences, cuts_text)
        else:
            print(path)
    word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    with open(word_freq_path, 'wb') as f:
        pickle.dump(word_freq, f)
    with open(sentences_path, 'wb') as f:
        pickle.dump(sentences, f)

    return sentences, word_freq


if __name__ == "__main__":
    sentences = cut_words()
    # print(sentences)
