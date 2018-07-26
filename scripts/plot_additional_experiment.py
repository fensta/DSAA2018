"""
Plot resulting AUC curves for additional experiment
"""
import os
import csv

import matplotlib.pyplot as plt
import numpy as np


FONTSIZE = 11
plt.rcParams.update({'font.size': FONTSIZE})
#Latex Backend for Matplotlib
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"


def plot_auc_curves(xs, ys, names, dst):
    """
    Plots AUC curves for multiple datasets.

    Parameters
    ----------
    xs: list of lists - each inner list represents x-values for a specific
    dataset.
    ys: list of lists - each inner list represents corresponding y-values for a
    specific dataset. Same order as <xs>.
    names: list of str - names of the datasets. Same order as in <xs> and <ys>.
    dst: str - path where plot will be stored.

    """
    fig = plt.figure(figsize=(5, 3))
    colors = ["darkorange", "dodgerblue", "black", "orchid", "lawngreen"]
    ax = fig.add_subplot(111)
    for idx, (x,y) in enumerate(zip(xs, ys)):
        avg_auc = sum(y) / len(y)
        ax.plot(x, y, color=colors[idx], label=names[idx] +
                " (AUC={:.2f})".format(avg_auc))
    # y-axis from 0-1
    plt.yticks(np.arange(0.4, 0.85, 0.1))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    ax.set_xlabel("\% of tweets with disagreement")
    ax.set_ylabel("AUC")

    # Add a legend
    ax.legend(loc="best", shadow=True, fontsize=9, ncol=2)

    plt.savefig(dst, bbox_inches="tight", dpi=600)


def read_from_csv(src_dir, tweets):
    """
    Reads x and y values from csv files. Values in those files must be comma-
    separated.
    Input format of csv files:
    #tweets with high disagreement in line 1
    AUC scores in line 2
    0,5,10,15...
    0.6, 0.76, 0.34...

    Parameters
    ----------
    src_dir: str - directory in which the csv files are stored.
    tweets: int - number of tweets in dataset.

    Returns
    -------
    list of lists, list of lists.
    First value represents x-values, second one the y-values. In each list
    the inner list represents the x (or y) values of a specific dataset.

    """
    xs = []
    ys = []
    for ds in os.listdir(src_dir):
        # Skip results from unrelated datasets
        if ds.startswith(str(tweets)):
            p = os.path.join(src_dir, ds)
            with open(p, "rb") as f:
                reader = csv.reader(f, delimiter=",", quotechar='"')
                x = []
                y = []
                for idx, row in enumerate(reader):
                        for val in row:
                            # x
                            if idx == 0:
                                # Convert to %
                                perc = 1.0 * float(val) / tweets * 100
                                x.append(perc)
                            # y
                            else:
                                y.append(float(val))
                xs.append(x)
                ys.append(y)
    return xs, ys


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    FIG_DIR = os.path.join(base_dir, "results", "figures")

    #########################################################
    # 1. Plot results obtained with optimized classifiers   #
    # Tweets with disagreement are up to 50% of the dataset #
    #########################################################

    # Directory in which the results are stored in csv files
    RES_DIR = os.path.join(base_dir, "results", "additional_experiment",
                           "results_optimized")
    # Number of tweets in dataset
    tweets = 174
    fname = "influence_low_agreement_on_classifier_opti_{}.pdf".format(tweets)
    dst = os.path.join(FIG_DIR, fname)
    # Names of the datasets
    names = ["4 votes", "5 votes", "6 votes", "7 votes", "8 votes"]
    xs, ys = read_from_csv(RES_DIR, tweets)
    plot_auc_curves(xs, ys, names, dst)

    ##########################################################
    # 2. Plot results obtained with optimized classifiers    #
    # Tweets with disagreement are up to 100% of the dataset #
    ##########################################################

    # Directory in which the results are stored in csv files
    RES_DIR = os.path.join(base_dir, "results", "additional_experiment",
                           "results_optimized")
    # Number of tweets in dataset
    tweets = 87
    fname = "influence_low_agreement_on_classifier_opti_{}.pdf".format(tweets)
    dst = os.path.join(FIG_DIR, fname)
    # Names of the datasets
    names = ["4 votes", "5 votes", "6 votes", "7 votes", "8 votes"]
    xs, ys = read_from_csv(RES_DIR, tweets)
    plot_auc_curves(xs, ys, names, dst)
