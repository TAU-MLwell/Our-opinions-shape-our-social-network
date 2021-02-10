import numpy as np
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.collections import PolyCollection
import matplotlib.pylab as pl
import seaborn as sns

def plot_all_symlog_loglog(x, ys, x_label, y_label, title, legend=[], legend_title=""):
    if legend == []:
        legend = x
    fig, ax = plt.subplots()
    colors = pl.cm.jet(np.linspace(0, 1, len(legend)))
    for i, y in enumerate(ys):
        plt.plot(x, y, color=colors[i])
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(legend, title=legend_title)
    plt.show()


def plot_all(x, ys, x_label, y_label, title, legend=[], legend_title=""):
    if legend == []:
        legend = x
    colors = pl.cm.jet(np.linspace(0, 1, len(legend)))
    sns.set_theme()
    for i, y in enumerate(ys):
        # plt.xticks(x)
        plt.plot(x, y, '.-', color=colors[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(legend, title=legend_title)
    plt.show()
    return


def plot_graph(x, y, x_label, y_label, title, legend=[], legend_title=""):
    if legend == []:
        legend = x
    # plt.xticks(x)
    # plt.yticks(y)
    sns.set_theme()
    plt.plot(x, y, '.-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend(legend, title=legend_title)
    plt.tight_layout()
    plt.show()
    return


def plot_mat(mat, x_label, y_label, title):
    plt.matshow(mat, cmap=plt.cm.Blues)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
