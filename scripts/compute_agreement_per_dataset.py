"""
Computes label agreement within each of the crowdsourced datasets.
Computes it on 2 levels: in terms of relevance and in terms of sentiment.
"""
import os
from collections import Counter
from statsmodels.stats.inter_rater import fleiss_kappa
import codecs
import json

# from fuzzywuzzy import process
import numpy as np
import unicodecsv as csv
import matplotlib.pyplot as plt

# from create_dataset_trec_agreement_task import read_dataset
# from add_tweet_ids_and_expert_labels_to_gizems_crowdsourced_dataset import \
#     read_gizem_json

# Since the AMT project was copied from another experiment, we forgot to
# change the names of the radio buttons in the new layout, so the old labels
# from the previous TREC dataset experiment
# (not relevant, relevant, highly relevant, I can't judge) were used
# Thus, we need to rename them manually.
LABEL_MAPPING = {
    "Not Relevant": "Positive",
    "Relevant": "Neutral",
    "Highly Relevant": "Negative",
    "I can't judge": "Irrelevant"
}

# Assigns each label an index in the table for Fleiss' Kappa computation
NUMBER_MAPPING = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
    "Irrelevant": 3
}

FONTSIZE = 11
plt.rcParams.update({'font.size': FONTSIZE})
#Latex Backend for Matplotlib
plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"


def compute_fleiss_kappa(data):
    """
    Computes label agreement between crowd workers according to Fleiss' Kappa
    w.r.t relevance of a tweet to the topic and sentiment separately.
    We have M rows representing tweets and N labels (from which a label was
    selected) as columns our matrix.
    Fleiss' Kappa is known to be a conservative metric as it sometimes yields
    low agreement, although the agreement is quite high in reality, see
    https://link.springer.com/article/10.1007/s11135-014-0003-1#page-1

    Parameters
    ----------
    data: dict - {tid: [label1, label2, label3...]}.

    Returns
    -------
    float, float.
    Fleiss' Kappa between 0 (no agreement) - 1 (perfect agreement) over all
    labels.
    Fleiss' Kappa between 0 (no agreement) - 1 (perfect agreement) investigating
    annotator agreement w.r.t. relevance only (irrelevant vs. rest).

    """
    # http://www.tau.ac.il/~tsirel/dump/Static/knowino.org/wiki/Fleiss%27_kappa.html
    # print """Agreement levels:
    #         < 0 	No agreement
    #         0.0 - 0.19 	Poor agreement
    #         0.20 - 0.39 	Fair agreement
    #         0.40 - 0.59 	Moderate agreement
    #         0.60 - 0.79 	Substantial agreement
    #         0.80 - 1.00 	Almost perfect agreement"""
    ############################
    # 1. Overall Fleiss' Kappa #
    ############################
    mat = np.zeros((len(data), len(LABEL_MAPPING)))
    # For each tweet
    for idx, tid in enumerate(data):
        labels = Counter(data[tid])
        # Count which labels exist for a tweet
        for label in labels:
            # Get the column of the label (= column to update)
            label_col = NUMBER_MAPPING[label]
            # Update the column with the votes of the crowd worers
            mat[idx, label_col] += labels[label]
    kappa_total = fleiss_kappa(mat)
    print "Overall Fleiss kappa:", kappa_total

    ########################################
    # 2. Fleiss' Kappa for tweet relevance #
    ########################################
    # We compare relevant (i.e. assigning a sentiment label) vs. irrelevant
    mat = np.zeros((len(data), 2))
    # Indices of the columns in the matrix for the two labels
    rel_col = 0
    irrel_col = 1
    # For each tweet
    for idx, tid in enumerate(data):
        labels = Counter(data[tid])
        # Count which labels exist for a tweet
        for label in labels:
            # Update the column with the votes of the crowd worers
            # a) Relevant
            if label != "Irrelevant":
                mat[idx, rel_col] += labels[label]
            # b) Irrelevant
            else:
                mat[idx, irrel_col] += labels[label]
    kappa_relevance = fleiss_kappa(mat)
    print "Relevance Fleiss kappa:", kappa_relevance
    return kappa_total, kappa_relevance

    # ########################################
    # # 3. Fleiss' Kappa for tweet sentiment #
    # ########################################
    # # We compare neutral, with negative and positive and ignore irrelevant
    # mat = np.zeros((len(data), 3))
    # # Indices of the columns in the matrix for the two labels
    #
    # # Assigns each label an index in the table for Fleiss' Kappa computation
    # SENT_MAPPING = {
    #     "Negative": 0,
    #     "Neutral": 1,
    #     "Positive": 2,
    # }
    # # For each tweet
    # for idx, tid in enumerate(data):
    #     labels = Counter(data[tid])
    #     # Count which labels exist for a tweet
    #     for label in labels:
    #         # Update the column with the votes of the crowd worers
    #         # Label is a sentiment label
    #         if label != "Irrelevant":
    #             label_col = SENT_MAPPING[label]
    #             mat[idx, label_col] += labels[label]
    # print "Sentiment Fleiss kappa:", fleiss_kappa(mat)


def read_amt_csv(src, votes, votes_per_tweet):
    """
    Reads in a csv file downloaded from AMT that contains the crowdsourced
    labels.

    Parameters
    ----------
    src: str - path to csv file.
    votes: int - number of annotator labels to consider. We always take the
    first <votes> labels.
    votes_per_tweet: int - specify how often each tweet was labeled in total.

    Returns
    -------
    dict.
    {tid: [label1, label2, label3, label]}

    """
    labels = {}
    with open(src, "rb") as f:
        reader = csv.reader(f)
        tweet_labels = []
        # Number of labels collected for the current tweet
        used_labels = 0
        for idx, row in enumerate(reader):
            # Skip header
            if idx > 0:
                if used_labels < votes:
                    label = row[-1]
                    label = LABEL_MAPPING[label]
                    tweet_labels.append(label)
                    used_labels += 1
                # Every <votes_per_tweet> times we extracted all labels of a
                # tweet
                if idx % votes_per_tweet == 0 and used_labels == votes:
                    tid = row[-4]
                    labels[tid] = tweet_labels
                    # Reset variables for next tweet
                    used_labels = 0
                    tweet_labels = []
    return labels


def use_n_labels(labels, n):
    """
    Select the first n labels.

    Parameters
    ----------
    labels: dict - {tid: [label1, label2, label3...]} list of labels per tweet.

    Returns
    -------
    dict.
    Dictionary holding only the first n labels per tweet.

    """
    return {tid: labels[tid][:n] for tid in labels}


def plot_label_distribution(counters, ds_names, title, fpath):
    """
    Plots a bar chart given some counted labels.

    Parameters
    ----------
    counters: list of collection.Counter - each counter represents a dataset
    and holds the raw counts of all labels that were assigned by crowd workers.
    ds_names: list of str - each string is the corresponding dataset name in
    <counters> and hence it uses the same order.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    # Order in which the bars are drawn from left to right
    xlabels = ["Positive", "Neutral", "Negative", "Irrelevant"]

    # Color for each experts/crowd
    COLORS = ["dodgerblue", "orangered", "black", "mediumorchid"]

    # 3 or 4 bars
    num_items = len(xlabels)
    print num_items, "bars"
    x = np.arange(num_items)
    print "x", x
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Transparency of bars
    opacity = 0.3
    # Width of a bar
    width = 0.95
    # Stores the maximum height of any of the bars - necessary for displaying
    # text over the bars at a certain x-value
    y_max_heights = np.zeros(num_items)

    # For each dataset separately
    for i, (counter, labl) in enumerate(zip(counters, ds_names)):
        y = []
        # Compute total number of labels because we want to display %
        # on y-axis
        total = sum(counter.values())

        # Use order defined in xlabels for plotting the bars
        for label in xlabels:
            prcnt = 1.0 * counter[label] / total * 100
            y.append(prcnt)
        # Use same color for all bars of expert/crowd
        for idx, y_ in enumerate(y):
            step_size = 0
            # Update height of a given bar
            if y_ > y_max_heights[idx]:
                y_max_heights[idx] = y_
            step_size += 0.5/((i+1))
            # https://stackoverflow.com/questions/23293011/how-to-plot-a-superimposed-bar-chart-using-matplotlib-in-python
            ax.bar(
                x[idx] + i*(step_size * width),    # later plots have x-offset as bars are thinner
                y_,
                label=xlabels[idx],
                color=COLORS[i],
                width=1.0*width / (i+1),     # 2nd plotted distribution has thinner bars
                alpha=opacity)

    ax.set_xticks(x)
    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])

    # Display labels on top of plotted bars
    rects = ax.patches

    for idx, (rect, label) in enumerate(zip(rects, xlabels)):
        # height = rect.get_height()
        # Height is specified in %
        # Get height of heighest bar at that x-value
        height = y_max_heights[idx]
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')

    # Rotate x-axis labels
    # ax.set_xticklabels(xlabels, rotation=45)
    # Hide x-axis labels
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    ax.set_xlabel("Sentiment labels")
    ax.set_ylabel("Percentage")

    # Add a legend
    ax.legend(ds_names,
              loc="upper left", shadow=True, fontsize=FONTSIZE)
    leg = ax.get_legend()
    # Set color of <labelers> according to plot
    for i in range(len(counters)):
        leg.legendHandles[i].set_color(COLORS[i])

    plt.savefig(fpath, bbox_inches="tight", dpi=600)


def plot_label_distribution_multiple(counters, ds_names, title, fpath):
    """
    Same as plot_label_distribution(), but plots each bar separately instead
    of having nested bars to improve readability.

    Parameters
    ----------
    counters: list of collection.Counter - each counter represents a dataset
    and holds the raw counts of all labels that were assigned by crowd workers.
    ds_names: list of str - each string is the corresponding dataset name in
    <counters> and hence it uses the same order.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    # Order in which the bars are drawn from left to right
    xlabels = ["Positive", "Neutral", "Negative", "Irrelevant"]

    # Color for each experts/crowd
    COLORS = ["dodgerblue", "orangered", "black", "mediumorchid"]
    print "####MULTIPLE####"
    # 3 or 4 bars
    num_items = len(xlabels)
    print num_items, "bars"
    x = np.arange(num_items)
    print "x", x
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Number of groups
    groups = len(counters)
    # Transparency of bars
    opacity = 0.3
    # Space between bars
    space = 0.2
    # Width of a bar
    width = (1 - space) / groups
    # Stores the maximum height of any of the bars - necessary for displaying
    # text over the bars at a certain x-value
    y_max_heights = np.zeros(num_items)

    # For each dataset separately
    for i, (counter, labl) in enumerate(zip(counters, ds_names)):
        y = []
        # Compute total number of labels because we want to display %
        # on y-axis
        total = sum(counter.values())

        # Use order defined in xlabels for plotting the bars
        for label in xlabels:
            prcnt = 1.0 * counter[label] / total * 100
            y.append(prcnt)
        # Use same color for all bars of expert/crowd
        for idx, y_ in enumerate(y):
            # Update height of a given bar
            if y_ > y_max_heights[idx]:
                y_max_heights[idx] = y_
            # https://stackoverflow.com/questions/23293011/how-to-plot-a-superimposed-bar-chart-using-matplotlib-in-python
            ax.bar(
                x[idx] + i*width,    # later plots have x-offset as bars are thinner
                y_,
                label=xlabels[idx],
                color=COLORS[i],
                width=1.0*width,     # 2nd plotted distribution has thinner bars
                alpha=opacity)

    ax.set_xticks(x)
    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])

    # Display labels on top of plotted bars
    rects = ax.patches

    for idx, (rect, label) in enumerate(zip(rects, xlabels)):
        # height = rect.get_height()
        # Height is specified in %
        # Get height of highest bar at that x-value
        height = y_max_heights[idx]
        # Center name over the group of bars
        ax.text(rect.get_x() + groups*rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')

    # Rotate x-axis labels
    # ax.set_xticklabels(xlabels, rotation=45)
    # Hide x-axis labels
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    ax.set_xlabel("Sentiment labels")
    ax.set_ylabel("Percentage")

    # Add a legend
    ax.legend(ds_names,
              loc="upper left", shadow=True, fontsize=FONTSIZE)
    leg = ax.get_legend()
    # Set color of <labelers> according to plot
    for i in range(len(counters)):
        leg.legendHandles[i].set_color(COLORS[i])

    plt.savefig(fpath, bbox_inches="tight", dpi=600)


def plot_agree_distribution(counters, ds_names, title, fpath):
    """
    Plots a bar chart given some counted labels.

    Parameters
    ----------
    counters: list of collection.Counter - each counter represents a dataset
    and holds the raw counts of all labels that were assigned by crowd workers.
    ds_names: list of str - each string is the corresponding dataset name in
    <counters> and hence it uses the same order.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    # Order in which the bars are drawn from left to right
    xlabels = ["disagreement", "no disagreement"]
    xlabels_names = ["low", "high"]

    # Color for each experts/crowd
    COLORS = ["dodgerblue", "orangered", "black", "mediumorchid"]

    # 3 or 4 bars
    num_items = len(xlabels)
    x = np.arange(num_items)
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Transparency of bars
    opacity = 0.3
    # Width of a bar
    width = 0.95
    # Stores the maximum height of any of the bars - necessary for displaying
    # text over the bars at a certain x-value
    y_max_heights = np.zeros(num_items)

    # For each dataset separately
    for i, (counter, labl) in enumerate(zip(counters, ds_names)):
        y = []
        # Compute total number of labels because we want to display %
        # on y-axis
        total = sum(counter.values())

        # Use order defined in xlabels for plotting the bars
        for label in xlabels_names:
            prcnt = 1.0 * counter[label] / total * 100
            y.append(prcnt)
        # Use same color for all bars of expert/crowd
        for idx, y_ in enumerate(y):
            step_size = 0
            # Update height of a given bar
            if y_ > y_max_heights[idx]:
                y_max_heights[idx] = y_
            step_size += 0.5/(i+1)
            # https://stackoverflow.com/questions/23293011/how-to-plot-a-superimposed-bar-chart-using-matplotlib-in-python
            ax.bar(
                x[idx] + i*(step_size * width),    # later plots have x-offset as bars are thinner
                y_,
                label=xlabels[idx],
                color=COLORS[i],
                width=1.0*width / (i+1),     # 2nd plotted distribution has thinner bars
                alpha=opacity)

    ax.set_xticks(x)
    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])

    # Display labels on top of plotted bars
    rects = ax.patches

    for idx, (rect, label) in enumerate(zip(rects, xlabels)):
        # height = rect.get_height()
        # Height is specified in %
        # Get height of heighest bar at that x-value
        height = y_max_heights[idx]
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')

    # Rotate x-axis labels
    # ax.set_xticklabels(xlabels, rotation=45)
    # Hide x-axis labels
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    ax.set_xlabel("Worker disagreement on sentiment labels")
    ax.set_ylabel("Percentage")

    # Add a legend
    ax.legend(ds_names,
              loc="upper left", shadow=True, fontsize=FONTSIZE)
    leg = ax.get_legend()
    # Set color of <labelers> according to plot
    for i in range(len(counters)):
        leg.legendHandles[i].set_color(COLORS[i])

    plt.savefig(fpath, bbox_inches="tight", dpi=600)


def plot_agree_distribution_multiple(counters, ds_names, title, fpath):
    """
    Same as plot_agree_distribution(), but plots bars separately instead of
    nested.

    Parameters
    ----------
    counters: list of collection.Counter - each counter represents a dataset
    and holds the raw counts of all labels that were assigned by crowd workers.
    ds_names: list of str - each string is the corresponding dataset name in
    <counters> and hence it uses the same order.
    title: str - title of the plot.
    fpath: str - path where the plot is stored.

    """
    # Since we display disagreement instead of agreement, the displayed
    # labels are inverted
    xlabels = ["disagreement", "no disagreement"]
    # Order in which the bars are drawn from left to right
    xlabels_names = ["low", "high"]

    # Color for each experts/crowd
    COLORS = ["dodgerblue", "orangered", "black", "mediumorchid"]

    # 3 or 4 bars
    num_items = len(xlabels)
    x = np.arange(num_items)
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Number of groups
    groups = len(counters)
    # Transparency of bars
    opacity = 0.3
    # Space between bars
    space = 0.2
    # Width of a bar
    width = (1 - space) / groups
    # Stores the maximum height of any of the bars - necessary for displaying
    # text over the bars at a certain x-value
    y_max_heights = np.zeros(num_items)

    # For each dataset separately
    for i, (counter, labl) in enumerate(zip(counters, ds_names)):
        y = []
        # Compute total number of labels because we want to display %
        # on y-axis
        total = sum(counter.values())

        # Use order defined in xlabels for plotting the bars
        for label in xlabels_names:
            prcnt = 1.0 * counter[label] / total * 100
            y.append(prcnt)
        # Use same color for all bars of expert/crowd
        for idx, y_ in enumerate(y):
            # Update height of a given bar
            if y_ > y_max_heights[idx]:
                y_max_heights[idx] = y_
            ax.bar(
                x[idx] + i*width,
                y_,
                label=xlabels[idx],
                color=COLORS[i],
                width=1.0*width,
                alpha=opacity)

    ax.set_xticks(x)
    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])

    # Display labels on top of plotted bars
    rects = ax.patches

    for idx, (rect, label) in enumerate(zip(rects, xlabels)):
        # height = rect.get_height()
        # Height is specified in %
        # Get height of heighest bar at that x-value
        height = y_max_heights[idx]
        # Center name over the group of bars
        ax.text(rect.get_x() + groups * rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')

    # Rotate x-axis labels
    # ax.set_xticklabels(xlabels, rotation=45)
    # Hide x-axis labels
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    ax.set_xlabel("Worker disagreement on sentiment labels")
    ax.set_ylabel("Percentage")

    # Add a legend
    ax.legend(ds_names,
              loc="upper left", shadow=True, fontsize=FONTSIZE)
    leg = ax.get_legend()
    # Set color of <labelers> according to plot
    for i in range(len(counters)):
        leg.legendHandles[i].set_color(COLORS[i])

    plt.savefig(fpath, bbox_inches="tight", dpi=600)


def analyze_tweet_distribution_variable_labels(low4, low8):
    """
    Analyzes to which agreement category tweets belong when using more labels.

    Parameters
    ----------
    low4: dict. - {tid: [l1, l2, l3]}
    low8:

    """
    low_ag4, _ = get_low_perfect_agreement(low4, 4)
    low_ag8, _ = get_low_perfect_agreement(low8, 8)
    high_ag4 = {}
    high_ag8 = {}
    # If a tweet has no low agreement, it has high agreement
    for tid in low4:
        if tid not in low_ag4:
            high_ag4[tid] = None
    for tid in low8:
        if tid not in low_ag8:
            high_ag8[tid] = None
    print "\nAgreement on LOW using 4 labels"
    print "------------------"
    print "low: {}/{} ({})".format(len(low_ag4), len(low4), 1.0*len(low_ag4) /
                                   len(low4))
    print "high: {}/{} ({})".format(len(high_ag4), len(low4), 1.0 *
                                    len(high_ag4) / len(low4)*100)

    print "\nAgreement on LOW using 8 labels"
    print "------------------"
    print "low: {}/{} ({})".format(len(low_ag8), len(low8), 1.0*len(low_ag8) /
                                   len(low8))
    print "high: {}/{} ({})".format(len(high_ag8), len(low8), 1.0 *
                                    len(high_ag8) / len(low8)*100)

    still_low = 0
    still_high = 0
    now_low = 0
    now_high = 0
    for tid in low_ag4:
        if tid in low_ag8:
            still_low += 1
        else:
            now_high += 1

    for tid in high_ag4:
        if tid in high_ag8:
            still_high += 1
        else:
            now_low += 1

    print "\nChanges:"
    print "----------"
    print "still low: {}/{} ({})".format(still_low, len(low_ag4),
                                         100.0*still_low / len(low_ag4))
    print "now high: {}/{} ({})".format(now_high, len(low_ag4),
                                        100.0 * now_high / len(low_ag4))
    print "still high: {}/{} ({})".format(still_high, len(high_ag4),
                                          100.0 * still_high / len(high_ag4))
    print "now low: {}/{} ({})".format(now_low, len(high_ag4),
                                       100.0 * now_low / len(high_ag4))


def get_low_perfect_agreement(dataset, votes):
    """
    Returns the number of low and perfect agreement tweets in the dataset.
    Low agreement is if <= 50% of the annotators assigned the majority label.
    For perfect agreement, all votes are identical.

    Parameters
    ----------
    dataset: dict - {tid: [label1, label2, label3]...]}
    {"label1": #occurrences in dataset)
    votes: int - number of votes per tweet. If a majority label got <= <votes>/2
    votes, it's considered low agreement.

    Returns
    -------
    dict, dict.
    {tid: (majority_label, #votes} - tweets with low agreement.
    {tid: (majority_label, #votes} - tweets with perfect agreement.

    """
    low_agree = {}
    perfect_agree = {}
    for tid in dataset:
        distrib = Counter(dataset[tid])
        majority, count = distrib.most_common()[0]
        # Low agreement
        if count <= 1.0*votes/2:
            low_agree[tid] = (majority, count)
        # Perfect agreement
        elif count == votes:
            perfect_agree[tid] = (majority, count)
    return low_agree, perfect_agree


def get_low_agreement(dataset, votes):
    """
    Returns the number of low and perfect agreement tweets in the dataset.
    Low agreement is if <= 50% of the annotators assigned the majority label.
    For perfect agreement, all votes are identical.

    Parameters
    ----------
    dataset: dict - {tid: [l1, l2, l3]}
    votes: int - number of votes per tweet. If a majority label got <= <votes>/2
    votes, it's considered low agreement.

    Returns
    -------
    collections.Counter.
    {"low": 5, "high": 3}

    """
    agree = []
    for tid in dataset:
        distrib = Counter(dataset[tid])
        majority, count = distrib.most_common()[0]
        # Low agreement
        if count <= 1.0*votes/2:
            agree.append("low")
        else:
            agree.append("high")
    return Counter(agree)


def get_line_with_slope(x, y_):
    m = -1.0 * (y_[0] - y_[-1]) / len(x)
    n = y_[0]-m*x[0]
    y = [m*i+n for i in x]
    return y


def plot_low_agreement_fraction(dicts, ds_names, max_votes, dst):
    """
    Plot percentage of low agreement tweets in a dataset w.r.t. number of
    labels used to determine majority labels.

    Parameters
    ----------
    dicts: list of dict - [{tid: [l1, l2, l3],... }, ..] each dict represents a
    dataset.
    ds_names: list of str - names of the datasets. Same order as <dicts>.
    max_votes: int - maximum number of votes to be considered in plot.
    From 2 until <max_votes> labels will be plotted.
    dst: str - path where plot will be stored.

    """
    # {
    #     <ds_name>: [list of computed % of low agreement tweets],
    # }
    ys = {}
    xs = {}
    for dsn in ds_names:
        ys[dsn] = []
        xs[dsn] = []
    # % of low agreement tweets available
    y_avg = []
    # Number of labels used for computing y
    x = []
    # Maximum number of labels available per dataset
    max_labels = []
    # Count and store how many votes exist per tweet in a dataset
    for ds in dicts:
        for idx, tid in enumerate(ds):
            # Store how many labels exist for each dataset
            if idx == 0:
                max_labels.append(len(ds[tid]))
                break

    # # For each number of labels
    # for num_labels in xrange(2, max_votes+1):
    #     # Average the agreement scores for <num_labels> labels over all datasets
    #     low_agree_for_num_labels = 0
    #     total_labels = 0
    #     for ds_idx, (ds, ds_name) in enumerate(zip(dicts, ds_names)):
    #         # If dataset has <num_labels> labels
    #         if num_labels <= max_labels[ds_idx]:
    #             labls = use_n_labels(ds, num_labels)
    #             # print "use {} labels: {}".format(num_labels, labls)
    #             low_agree, _ = get_low_perfect_agreement(labls, num_labels)
    #             low_agree_for_num_labels += len(low_agree)
    #             total_labels += len(ds)
    #             frac = 1.0 * len(low_agree) / len(ds) * 100
    #             ys[ds_name].append(frac)
    #
    #     # Store % of low agreement tweets over all datasets
    #     frac = 1.0*low_agree_for_num_labels / total_labels * 100
    #     y_avg.append((frac))
    #     x.append(num_labels)
    #
    # # Plotting
    # fig = plt.figure(figsize=(5, 3))
    # ax = fig.add_subplot(111)
    # # Color for each experts/crowd
    # COLORS = ["dodgerblue", "orangered", "darkorchid"]
    # # Add average
    # ax.plot(x, y_avg, color=COLORS[1], label="Average")
    # # for idx, ds_name in enumerate(ys):
    # #     num_labels = len(ys[ds_name])
    # #     # +2 because we start with 2 instead of 0 labels
    # #     ax.plot([i+2 for i in range(num_labels)], ys[ds_name],
    # #             color=COLORS[idx], label=ds_name)
    #
    # ax.plot(x, get_line_with_slope(x, y_avg), "--", color="black")

    # For each number of labels
    for num_labels in xrange(2, max_votes + 1):
        # Average the agreement scores for <num_labels> labels over all datasets
        low_agree_for_num_labels = 0
        total_labels = 0
        # For each dataset
        for ds_idx, (ds, ds_name) in enumerate(zip(dicts, ds_names)):
            # If dataset has <num_labels> labels
            if num_labels <= max_labels[ds_idx]:
                labls = use_n_labels(ds, num_labels)
                # print "use {} labels: {}".format(num_labels, labls)
                low_agree, _ = get_low_perfect_agreement(labls, num_labels)
                #low_agree_for_num_labels += len(low_agree)
                frac = 1.0 * len(low_agree) / len(ds) * 100
                ys[ds_name].append(frac)
                xs[ds_name].append(num_labels)

        # Store % of low agreement tweets over all datasets
        # frac = 1.0 * low_agree_for_num_labels / total_labels * 100
        # y_avg.append((frac))
        # x.append(num_labels)

    # Plotting
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Color for each experts/crowd
    COLORS = ["dodgerblue", "orangered", "black", "mediumorchid"]
    for idx, ds_name in enumerate(xs):
        ax.plot(xs[ds_name], ys[ds_name], color=COLORS[idx], label=ds_name)
        #ax.plot(xs[ds_name], get_line_with_slope(x, y_avg), "--", color="black")



    # plt.title(title)
    # Set title position
    ttl = ax.title
    ttl.set_position([.5, 1.05])

    # y-axis from 0-100
    plt.yticks(np.arange(0, 110, 10))
    # Hide the right and top spines (lines)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    # Set labels of axes
    ax.set_xlabel("Votes per tweet")
    ax.set_ylabel("Disagreement in \%")

    # Add a legend
    ax.legend(loc="best", shadow=True, fontsize=FONTSIZE)
    plt.savefig(dst, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Directory in which the AMT results are stored
    DS_DIR = os.path.join(base_dir, "results", "dataset_twitter_crowdsourcing")
    # Directory in which the figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures")

    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    ############################
    # 1. Compute Fleiss' Kappa #
    ############################
    print "FLEISS KAPPA"
    print "-------------\n"
    print "HIGH"
    fname = "Batch_2984071_batch_results_high_4000.csv"
    src = os.path.join(DS_DIR, fname)
    votes = 4
    first_n = 4
    high = read_amt_csv(src, first_n, votes)
    compute_fleiss_kappa(high)

    print "MEDIUM"
    print "------"
    fname = "Batch_2984078_batch_results_medium_4000.csv"
    src = os.path.join(DS_DIR, fname)
    votes = 4
    first_n = 4
    medium = read_amt_csv(src, first_n, votes)
    compute_fleiss_kappa(medium)

    print "LOW"
    print "---"
    # Check if number of votes improves agreement
    fname = "Batch_2984090_batch_results_low_8000.csv"
    src = os.path.join(DS_DIR, fname)
    votes = 8
    first_n = 8
    low = read_amt_csv(src, first_n, votes)
    # a) Using 4 labels
    low_four = use_n_labels(low, 4)
    four_all, four_relevance = compute_fleiss_kappa(low_four)
    print "using 4 labels", four_all, four_relevance

    # b) Using 8 labels
    low_eight = low
    votes = 8
    first_n = 8
    # low_eight = read_amt_csv(src, first_n, votes)
    low_eight = use_n_labels(low, 8)
    eight_all, eight_relevance = compute_fleiss_kappa(low_eight)
    print "using 8 labels", eight_all, eight_relevance
    effect_all = "improves"
    effect_relevance = "improves"
    if four_all >= eight_all:
        effect_all = "reduces"
    if four_relevance >= eight_relevance:
        effect_relevance = "reduces"
    print "Using 8 votes instead of 4 {} the overall inter-annotator " \
          "agreement by {}".format(effect_all, eight_all-four_all)
    print "Using 8 votes instead of 4 {} the overall inter-annotator " \
          "agreement by {}".format(effect_relevance, eight_relevance -
                                   four_relevance)

    # Count labels per dataset
    low_labels = []
    for tid in low_four:
        low_labels.extend(low_four[tid])
    med_labels = []
    for tid in medium:
        med_labels.extend(medium[tid])
    hig_labels = []
    for tid in high:
        hig_labels.extend(high[tid])
    low_labels_eight = []
    for tid in low_eight:
        low_labels_eight.extend(low_eight[tid])

    low_labels = Counter(low_labels)
    med_labels = Counter(med_labels)
    hig_labels = Counter(hig_labels)
    low_labels_eight = Counter(low_labels_eight)
    print "distribution using 4 labels", low_labels
    print "distribution using 8 labels", low_labels_eight

    train = os.path.join(base_dir, "results",
                         "dataset_twitter_agreement_experiment_fixed",
                         "twitter_agreement_train.json")
    # Load training dataset
    with codecs.open(train, "rb", encoding="utf-8") as f:
        train_tweets = json.load(f, encoding="utf-8")

    # Add training set
    train_labels = []
    for tid in train_tweets:
        sent_label = train_tweets[tid]["relevance_label"]
        if sent_label != "Irrelevant":
            sent_label = train_tweets[tid]["sentiment_label"]
        train_labels.append(sent_label)
    print "train labels", len(train_labels)
    train_labels = Counter(train_labels)

    ##########################################################################
    # Careful: in the code we used agreement, but in the paper disagreement,
    # so low agreement in the code corresponds to high disagreement and hig
    # agreement to low disagreement. That means LOW corresponds to low
    # disagreement and HIGH corresponds to high disagreement.
    ##########################################################################

    # Plot label distributions of all datasets using 4 labels
    datasets = [low_labels, med_labels, hig_labels, train_labels]
    # Switch order of labels since we now talk about disagreement instead of
    # agreement
    names = ["\\textsc{HIGH}", "\\textsc{MEDIUM}", "\\textsc{LOW}", "$S_0$"]
    fname = "crowdsourcing_label_distribution_4_labels.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_label_distribution(datasets, names, "", dst)
    # Same plot, but now plot multiple bars instead of nested ones
    fname = "crowdsourcing_label_distribution_4_labels_multiple.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_label_distribution_multiple(datasets, names, "", dst)

    # Plot label distribution for low agreement using 4 or 8 labels
    datasets = [low_labels, low_labels_eight]
    names = ["4 labels", "8 labels"]
    fname = "crowdsourcing_label_distribution_8_labels.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_label_distribution(datasets, names, "", dst)
    # Same plot, but now plot multiple bars instead of nested ones
    fname = "crowdsourcing_label_distribution_8_labels_multiple.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_label_distribution_multiple(datasets, names, "", dst)

    # Plot agreement label distribution using 8 labels for LOW and 4 labels
    # for MEDIUM, HIGH, and TRAIN
    agree_l = get_low_agreement(low, 8)
    agree_m = get_low_agreement(medium, 4)
    agree_h = get_low_agreement(high, 4)

    # Add training set
    train_labels = []
    for tid in train_tweets:
        train_labels.append(train_tweets[tid]["agreement"])
    print "train labels", len(train_labels)
    train_labels = Counter(train_labels)
    agree_t = Counter(train_labels)
    datasets = [agree_l, agree_m, agree_h, agree_t]
    # Switch order of labels since we now talk about disagreement instead of
    # agreement
    names = ["\\textsc{HIGH}", "\\textsc{MEDIUM}", "\\textsc{LOW}", "$S_0$"]
    fname = "crowdsourcing_agreement_distribution_4_labels.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_agree_distribution(datasets, names, "", dst)
    # Same plot, but now plot multiple bars instead of nested ones
    fname = "crowdsourcing_agreement_distribution_4_labels_multiple.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_agree_distribution_multiple(datasets, names, "", dst)

    # See how agreement develops in LOW if number of labels is increased from
    # 4 to 8
    agree_l_low = get_low_agreement(low_four, 4)
    datasets = [agree_l_low, agree_l]
    names = ["4 labels", "8 labels"]
    fname = "crowdsourcing_agreement_distribution_8_labels.pdf"
    dst = os.path.join(FIG_DIR, fname)
    print "tweets with low agreement using 4 labels"
    print agree_l_low["low"]
    print "tweets with low agreement using 8 labels"
    print agree_l["low"]
    plot_agree_distribution(datasets, names, "", dst)
    # Same plot, but now plot multiple bars instead of nested ones
    fname = "crowdsourcing_agreement_distribution_8_labels_multiple.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_agree_distribution_multiple(datasets, names, "", dst)

    # See if agreement improves with more labels if it's initially low (<= 50%
    # votes for majority label)
    low_agree_after_four, perfect_agree_after_four = \
        get_low_perfect_agreement(low_four, 4)
    total = len(low_four)
    la = 1.0*len(low_agree_after_four) / total * 100
    between = {}
    # Could be medium or high agreement tweets
    for tid in low_four:
        if tid not in low_agree_after_four \
                and tid not in perfect_agree_after_four:
            between[tid] = None
    middle = 1.0*len(between) / total * 100
    perfect = 1.0*len(perfect_agree_after_four) / total * 100
    print "\nLOW using 4 labels"
    print "---------------"
    print "#low agreement: {}/{} ({}%)".format(len(low_agree_after_four),
                                               total, la)
    print "#between agreement: {}/{} ({}%)".format(len(between), total, middle)
    print "#perfect agreement: {}/{} ({}%)"\
        .format(len(perfect_agree_after_four), total, perfect)

    has_majority_now = {}
    still_low = {}
    # Now see if agreement improved after requesting 8 labels
    for tid in low_agree_after_four:
        distrib = Counter(low_eight[tid])
        majority, count = distrib.most_common()[0]
        # It's still low agreement
        if count < 5:
            still_low[tid] = (majority, count)
        # Has now high agreement
        else:
            has_majority_now[tid] = (majority, count)

    low_agree_after_eight, perfect_agree_after_eight = \
        get_low_perfect_agreement(low_eight, 8)
    total = len(low_eight)
    la = 1.0 * len(low_agree_after_eight) / total * 100
    between = {}
    # Could be medium or high agreement tweets
    for tid in low_eight:
        if tid not in low_agree_after_eight \
                and tid not in perfect_agree_after_eight:
            between[tid] = None
    middle = 1.0 * len(between) / total * 100
    perfect = 1.0 * len(perfect_agree_after_eight) / total * 100

    print "\nLOW using 8 labels"
    print "---------------"
    print "#low agreement: {}/{} ({}%)".format(len(low_agree_after_eight),
                                               total, la)
    print "#between agreement: {}/{} ({}%)".format(len(between), total, middle)
    print "#perfect agreement: {}/{} ({}%)" \
        .format(len(perfect_agree_after_eight), total, perfect)
    print "#improved agreement", len(has_majority_now)
    print "#still low agreement", len(still_low)

    still_perfect = {}
    still_high = {}
    now_low = {}
    # Now see if agreement improved after requesting 8 labels
    for tid in perfect_agree_after_four:
        distrib = Counter(low_eight[tid])
        majority, count = distrib.most_common()[0]
        # Has now low agreement
        if count < 5:
            now_low[tid] = (majority, count)
        # Has still perfect agreement
        elif count == 8:
           still_perfect[tid] = (majority, count)
        # Has still high agreement
        else:
            still_high[tid] = (majority, count)
    print "#now low agreement", len(now_low)
    for tid in now_low:
        print tid
    print "#still high agreement", len(still_high)
    print "#still perfect agreement", len(still_perfect)

    # Check agreement for medium
    low_agree, perfect_agree = get_low_perfect_agreement(medium, 4)
    total = len(medium)
    la = 1.0 * len(low_agree) / total * 100
    between = {}
    # Could be medium or high agreement tweets
    for tid in medium:
        if tid not in low_agree \
                and tid not in perfect_agree:
            between[tid] = None
    middle = 1.0 * len(between) / total * 100
    perfect = 1.0 * len(perfect_agree) / total * 100
    print "\nMEDIUM"
    print "---------------"
    print "#low agreement: {}/{} ({}%)".format(len(low_agree), total, la)
    print "#between agreement: {}/{} ({}%)".format(len(between), total, middle)
    print "#perfect agreement: {}/{} ({}%)" \
        .format(len(perfect_agree), total, perfect)

    # Check disagreement for high agreement dataset
    low_agree, perfect_agree = get_low_perfect_agreement(high, 4)
    print "#low agreement", len(low_agree)
    print "#perfect agreement", len(perfect_agree)

    total = len(high)
    la = 1.0 * len(low_agree) / total * 100
    between = {}
    # Could be medium or high agreement tweets
    for tid in high:
        if tid not in low_agree \
                and tid not in perfect_agree:
            between[tid] = None
    middle = 1.0 * len(between) / total * 100
    perfect = 1.0 * len(perfect_agree) / total * 100

    print "\nHIGH"
    print "---------------"
    print "#low agreement: {}/{} ({}%)".format(len(low_agree),
                                               total, la)
    print "#between agreement: {}/{} ({}%)".format(len(between), total, middle)
    print "#perfect agreement: {}/{} ({}%)" \
        .format(len(perfect_agree), total, perfect)
    # low dataset has "low agreement", i.e. "high disagreement"
    datasets = [low, medium, high]
    ds_names = ["HIGH", "MEDIUM", "LOW"]
    max_votes = 8
    fname = "crowdsourcing_low_agreement_over_all_datasets.pdf"
    dst = os.path.join(FIG_DIR, fname)
    plot_low_agreement_fraction(datasets, ds_names, max_votes, dst)

    analyze_tweet_distribution_variable_labels(low_four, low_eight)
