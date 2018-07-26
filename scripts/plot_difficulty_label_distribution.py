"""
Plot Gizem's experimental results
"""
import os
import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

MAN_DIFFICULTY = "man_diff_label"
DEF_DIFFICULTY = "disagreement_diff_label"

FONTSIZE = 11
plt.rcParams.update({'font.size': FONTSIZE})
#Latex Backend for Matplotlib
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"


def plot_distribution(y, labels, title, fpath):
    """
    Plots a bar chart for the label difficulty distribution.

    Parameters
    ----------
    y: List - y-values to be plotted.
    labels: List of str - list of labels for x-axis displayed in the legend.
    Has the same order as <y>.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    # Number of labels
    num_items = len(labels)
    # Color for each annotator group S, M, L
    COLORS = ["dodgerblue", "orangered", "black", "mediumorchid", "yellowgreen"]
    # Bar graphs expect a total width of "1.0" per group
    # Thus, you should make the sum of the two margins
    # plus the sum of the width for each entry equal 1.0.
    # One way of doing that is shown below. You can make
    # The margins smaller if they're still too big.
    # See
    # http://stackoverflow.com/questions/11597785/setting-spacing-between-grouped-bar-plots-in-matplotlib
    # to calculate gap in between bars; if gap isn't large enough, increase
    # <margin
    margin = 0.1
    # width = (1.-0.5*margin) / num_items
    width = (1.-0.5*margin) / (num_items-3)
    # Transparency of bars
    opacity = 0.3
    ind = np.arange(num_items)
    x = margin + ind
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    # Convert to percentages
    y_ = []
    total = sum(y)
    for el in y:
        y_.append(1.0 * el / total * 100)

    # Print bars separately
    for idx, l in enumerate(labels):
        # Plot each bar separately
        ax.bar(x[idx], y_[idx], width, label=labels[idx], color=COLORS[idx],
               alpha=opacity)

    # Set labels for ticks
    #ax.set_xticks(ind)
    # Hide x-axis labels
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    # Rotate x-axis labels
    #ax.set_xticklabels(xlabels, rotation=45)
    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Hide x-axis ticks
    plt.xticks([], [])
    # Set labels of axes
    # ax.set_xlabel("Difficulty label distribution")
    ax.set_xlabel("Distribution of indicators for worker disagreement")
    ax.set_ylabel("Percentage")
    # Add a legend
    ax.legend(loc="best", shadow=True, fontsize=FONTSIZE)
    plt.savefig(fpath, bbox_inches="tight", dpi=600)


def read_excel_tweets(src):
    """
    Reads in tweets that were manually labeled w.r.t. difficulty in Excel.

    Parameters
    ----------
    src: str - path where csv file is stored.

    Returns
    -------
    dict.
    Tweets.
    {tid:
        {
            <MAN_DIFFICULTY>: manual difficulty label,  # {A, E, I, B, O}
            <DEF_DIFFICULTY>: definition difficulty label,  # {LD, HD}
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=";", dialect="excel")
        tweets = {}
        uniq = set()
        for idx, row in enumerate(reader):
            # tid, text, majority label, definition difficulty, manual
            # difficulty, explanation
            tid, _, _, def_diff, man_diff, explain = \
                row[0], row[1], row[2], row[3], row[4], row[5]
            # print tid, def_diff, man_diff
            # Some labels have whitespaces
            man_diff = man_diff.strip()
            # Sanity check - label exists for each tweet
            assert(len(man_diff) > 0)
            assert(len(def_diff) == 2)
            # Store info for tweet
            tweets[tid] = {}
            tweets[tid][MAN_DIFFICULTY] = man_diff
            tweets[tid][DEF_DIFFICULTY] = def_diff
            uniq.add(man_diff)
    return tweets


def count_labels(tweets, transl):
    """
    Count how often each label was used.

    Parameters
    ----------
    tweets: dict - tweets with labels.
    transl: dict - labels to be displayed in plot instead of Excel labels.

    Returns
    -------
    List, list.
    First list contains counts, second one the corresponding labels for the
    legend.
    """
    counts = []
    for tid in tweets:
        counts.append(tweets[tid][MAN_DIFFICULTY])

    data = Counter(counts)
    y = []
    labels = []
    for lab in data:
        # Get label to be displayed
        labels.append(transl[lab])
        y.append(data[lab])
    return y, labels


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    FIG_DIR = os.path.join(base_dir, "results", "figures")

    # Translate excel labels to those shown on plot
    conv = {
        "E": "(S)implicity",
        "A": "(A)mbiguity",
        "B": "(B)ackground",
        "I": "(I)rrelevance",
        "O": "(O)ther",
    }

    ################################################################
    # 1. Plot difficulty label distribution in HIGH using 5 labels #
    ################################################################
    fname = "difficulty_label_distribution_high_5_labels_full_names.pdf"
    dst = os.path.join(FIG_DIR, fname)
    # Without Easy
    # y = [74, 243, 56, 71]
    # With Easy
    y = [74, 243, 56, 44, 27]
    labels = ["(A)mbiguity", "(B)ackground", "(I)rrelevance", "(O)ther",
              "(S)implicity"]
    plot_distribution(y, labels, "", dst)

    ########################################################################
    # 2. Plot difficulty label distribution in TRAIN + LOW + MEDIUM + HIGH #
    # using 5 labels                                                       #
    ########################################################################
    fname = "difficulty_label_distribution_all_5_labels_full_names.pdf"
    src = os.path.join(base_dir, "results", "ambiguous_difficult",
                       "difficulty_labels_all_votes_correct_labeled.csv")
    dst = os.path.join(FIG_DIR, fname)
    tweets = read_excel_tweets(src)
    y, labels = count_labels(tweets, conv)
    plot_distribution(y, labels, "", dst)
