"""
Plot Gizem's experimental results
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


def plot_auc_curve(x, y, dst):
    """
    Plots one AUC curve.

    Parameters
    ----------
    x: list - x-values for a specific dataset.
    y: list - corresponding y-values for a specific dataset. Same order as <xs>.
    dst: str - path where plot will be stored.

    """
    fig = plt.figure(figsize=(5, 3))
    colors = ["darkorange", "dodgerblue", "black", "orchid", "green"]
    ax = fig.add_subplot(111)
    ax.plot(x, y, color=colors[0])
    y_hor = [y[0] for _ in range(len(x))]
    ax.plot(x, y_hor, "--", color="black")
    # y-axis from 0-1
    plt.yticks(np.arange(0, 1.05, 0.1))
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
    # ax.legend(loc="best", shadow=True, fontsize=9, ncol=2)

    plt.savefig(dst, bbox_inches="tight", dpi=600)


def read_from_csv(src, tweets):
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
    src: str - path to the csv file.
    tweets: int - number of tweets in dataset.

    Returns
    -------
    list, list.
    First value represents x-values, second one the y-values.

    """
    x = []
    y = []
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
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
    return x, y


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    FIG_DIR = os.path.join(base_dir, "results", "figures")

    #########################################################
    # 1. Plot results obtained with optimized classifiers   #
    # Tweets with disagreement are up to 50% of the dataset #
    #########################################################

    # Number of tweets in dataset
    tweets = 2200
    # Directory in which the results are stored in csv files
    res = os.path.join(base_dir, "results", "dataset_gizem",
                       "results_optimized", "{}.csv".format(tweets))

    fname = "influence_low_agreement_on_classifier_opti_{}.pdf".format(tweets)
    dst = os.path.join(FIG_DIR, fname)
    x, y = read_from_csv(res, tweets)
    plot_auc_curve(x, y, dst)

    ##########################################################
    # 2. Plot results obtained with optimized classifiers    #
    # Tweets with disagreement are up to 100% of the dataset #
    ##########################################################

    # Number of tweets in dataset
    tweets = 1100
    # Directory in which the results are stored in csv files
    res = os.path.join(base_dir, "results", "dataset_gizem",
                       "results_optimized", "{}.csv".format(tweets))

    fname = "influence_low_agreement_on_classifier_opti_{}.pdf".format(tweets)
    dst = os.path.join(FIG_DIR, fname)
    # Names of the datasets
    x, y = read_from_csv(res, tweets)
    plot_auc_curve(x, y, dst)
