import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D


def plt_3d(X_tsne, names):
    myfont = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
    ax = plt.figure(figsize=(14, 8)).add_subplot(111, projection='3d')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2])

    # print(X_tsne)
    for i in range(len(X_tsne)):
        x = X_tsne[i][0]
        y = X_tsne[i][1]
        z = X_tsne[i][2]
        ax.text(x, y, z, names[i], fontproperties=myfont)
    plt.show()


def plt_2d(X_tsne, high_freq_words):
    print(high_freq_words)
    plt.figure(figsize=(14, 8))
    myfont = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    for i in range(len(X_tsne)):
        x = X_tsne[i][0]
        y = X_tsne[i][1]
        plt.text(x, y, high_freq_words[i], fontproperties=myfont, size=12)
    plt.show()
