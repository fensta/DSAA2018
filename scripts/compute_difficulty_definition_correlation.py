"""
Computes the correlation between both definitions of difficulty:
a) derived from worker disagreement (<=50% of votes are majority label)
b) difficulty labels assigned by 2 authors (authors had to agree on labels)

In b) labels were A, B, I, O, and E, where the first 4 indicate tweets that are
difficult to label and the last one that tweets were straightforward to label.
"""
import os
import unicodecsv as csv
from collections import Counter

import scipy.stats as ss
import numpy as np
import pandas as pd


# Default label assigned by annotator - it indicates no label was assigned
# for a certain hierarchy level, e.g. ["Relevant", "Factual", "?"] means
# on 1st level "Relevant" was assigned, on 2nd level "Factual", hence the
# 3rd level wasn't displayed anymore, so "?" is used
EMPTY = "?"
# Default value for annotation time
ZERO = 0

# Keys for dictionary
TEXT = "text"
MAJORITY_LABEL = "maj_label"
DISAGREEMENT = "disag"
#MAN_DIFFICULTY = "man_diff_label"
GROUND_TRUTH_DIFF = "man_diff_label"
DEF_DIFFICULTY = "disagreement_diff_label"
EXPLANATION = "explain"
LABEL = "label"

TIME = "time"
TIMES = "times"
TID = "tid"
UID = "uid"
WID = "wid"
VOTES = "votes"
ORDERED = "ordered"
LABELED = "labeled"
TYPE = "ds_type"


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
            <GROUND_TRUTH_DIFF>: manual difficulty label,  # {A, E, I, B, O}
            <DEF_DIFFICULTY>: definition difficulty label,  # {LD, HD}
            <EXPLANATION>: explanation of <DIFFICULTY>
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
            tweets[tid][EXPLANATION] = explain
            tweets[tid][GROUND_TRUTH_DIFF] = man_diff
            tweets[tid][DEF_DIFFICULTY] = def_diff
            uniq.add(man_diff)
    return tweets


def compute_pearson_r(tweets):
    """
    Computes correlation (Pearson's r) between difficulty labels to identify
    a possible linear relationship.

    Convert manual labels (A, B, E, I, O) into LD (low disagreement) and HD
    (high disagreement) as follows:
    A, B, I, O -> HD
    E -> LD

    Parameters
    ----------
    tweets: dict - tweets {tid: {<MAN_DIFFICULTY>:..., <DEF_DIFFICULTY:...>}}

    Returns
    -------

    """
    # Convert labels to numbers
    DEF_2_NUM = {
        "HD": 0,
        "LD": 1
    }

    # Convert 5 labels to 2 labels
    MAN_2_NUM = {
        "A": "HD",
        "B": "HD",
        "I": "HD",
        "O": "HD",
        "E": "LD",
    }

    tids = []

    # Store all labels
    def_labels = []
    man_labels = []

    # For LD and HD
    for tid in tweets:
        tids.append(tid)
        man_label = DEF_2_NUM[MAN_2_NUM[tweets[tid][GROUND_TRUTH_DIFF]]]
        def_label = DEF_2_NUM[tweets[tid][DEF_DIFFICULTY]]
        # print "Man: {} {} -> {}".format(tid, tweets[tid][MAN_DIFFICULTY],
        #                                 man_label)
        # print "Def: {} {} -> {}".format(tid, tweets[tid][DEF_DIFFICULTY],
        #                                 def_label)

        man_labels.append(man_label)
        def_labels.append(def_label)

    x_all = np.array(def_labels)
    y_all = np.array(man_labels)
    print "Pearson's r and 2-tailed significance:"
    print "For all tweets:", ss.pearsonr(x_all, y_all)


def cramers_corrected_stat(confusion_matrix):
    """
    Calculate Cramers V statistic for categorical-categorical association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328

    Taken from:
    https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C
    3%A9rs-coefficient-matrix/39266194

    Parameters
    ----------
    confusion_matrix: pandas.DataFrame

    Returns
    --------
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0., phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


def compute_cramers_v(tweets):
    """
    Computes correlation (Cramer's V) between difficulty labels to identify
    a possible linear relationship.

    Convert manual labels (A, B, E, I, O) into LD (low disagreement) and HD
    (high disagreement) as follows:
    A, B, I, O -> HD
    E -> LD

    Parameters
    ----------
    tweets: dict - tweets {tid: {<MAN_DIFFICULTY>:..., <DEF_DIFFICULTY:...>}}

    Returns
    -------

    """
    # Convert 5 labels to 2 labels
    # MAN_2_NUM = {
    #     "A": "HD",
    #     "B": "HD",
    #     "I": "HD",
    #     "O": "HD",
    #     "E": "LD",
    # }

    # Store all labels
    def_labels = []
    man_labels = []
    tids = []

    # For LD and HD
    for tid in tweets:
        tids.append(tid)
        # man_label = MAN_2_NUM[tweets[tid][MAN_DIFFICULTY]]
        def_label = tweets[tid][DEF_DIFFICULTY]
        man_label = tweets[tid][GROUND_TRUTH_DIFF]
        # print "Man: {} {} -> {}".format(tid, tweets[tid][MAN_DIFFICULTY],
        #                                 man_label)
        # print "Def: {} {} -> {}".format(tid, tweets[tid][DEF_DIFFICULTY],
        #                                 def_label)
        man_labels.append(man_label)
        def_labels.append(def_label)

    data_all = {
        DEF_DIFFICULTY: def_labels,
        GROUND_TRUTH_DIFF: man_labels
    }

    df_all = pd.DataFrame(index=tids, data=data_all)
    # print df_all
    confusion_matrix_all = pd.crosstab(df_all[GROUND_TRUTH_DIFF],
                                       df_all[DEF_DIFFICULTY])
    # Output
    # [3500 rows x 2 columns]
    # disagreement_diff_label   HD    LD
    # man_diff_label
    # HD                       779   295
    # LD                       327  2099
    # 0.58514046244
    # Interpretation: 2099/2394 tweets with LD are the same and 295 are
    # considered as HD

    # disagreement_diff_label   HD    LD
    # man_diff_label
    # A                        213    94
    # B                        226   107
    # E                        327  2099
    # I                        258    47
    # O                         82    47
    # 0.593096041147
    # Interpretation of Cramer's V:
    # http://groups.chass.utoronto.ca/pol242/Labs/LM-3A/LM-3A_content.htm
    #LEVEL OF ASSOCIATION 	Verbal Description 	COMMENTS
    # 0.00 	No Relationship 	Knowing the independent variable does not help in predicting the dependent variable.
    # .00 to .15 	Very Weak 	Not generally acceptable
    # .15 to .20 	 Weak  	    Minimally acceptable
    # .20 to .25 	Moderate  	Acceptable
    # .25 to .30 	Moderately Strong  	Desirable
    # .30 to .35 	Strong  	Very Desirable
    # .35 to .40 	Very Strong 	Extremely Desirable
    # .40 to .50 	Worrisomely Strong 	Either an extremely good relationship or the two variables are measuring the same concept
    # .50 to .99 	Redundant 	The two variables are probably measuring the same concept.
    # 1.00 	Perfect Relationship.  	If we the know the independent variable, we can perfectly predict the dependent variable.
    print confusion_matrix_all
    print cramers_corrected_stat(confusion_matrix_all)


def create_confusion_matrix(tweets):
    """
    Create confusion matrix.

    Convert manual labels G (A, B, E, I, O) into LD (low disagreement) and HD
    (high disagreement) as follows:
    A, B, I, O -> HD
    E -> LD

    Parameters
    ----------
    tweets: dict - tweets {tid: {<MAN_DIFFICULTY>:..., <DEF_DIFFICULTY:...>}}

    """
    # Convert 5 labels to 2 labels
    MAN_2_NUM = {
        "A": "HD",
        "B": "HD",
        "I": "HD",
        "O": "HD",
        "E": "LD",
    }

    # Store all labels
    def_labels = []
    man_labels = []
    tids = []

    # For LD and HD
    for tid in tweets:
        tids.append(tid)
        man_label = MAN_2_NUM[tweets[tid][GROUND_TRUTH_DIFF]]
        def_label = tweets[tid][DEF_DIFFICULTY]
        # print "Man: {} {} -> {}".format(tid, tweets[tid][MAN_DIFFICULTY],
        #                                 man_label)
        # print "Def: {} {} -> {}".format(tid, tweets[tid][DEF_DIFFICULTY],
        #                                 def_label)
        man_labels.append(man_label)
        def_labels.append(def_label)

    data_all = {
        DEF_DIFFICULTY: def_labels,
        GROUND_TRUTH_DIFF: man_labels
    }

    df_all = pd.DataFrame(index=tids, data=data_all)
    # print df_all
    confusion_matrix_all = pd.crosstab(df_all[GROUND_TRUTH_DIFF],
                                       df_all[DEF_DIFFICULTY])
    # Output
    # [3500 rows x 2 columns]
    # disagreement_diff_label   HD    LD
    # man_diff_label
    # HD                       779   295
    # LD                       327  2099
    # 0.58514046244
    # Interpretation: 2099/2394 tweets with LD are the same and 295 are
    # considered as HD
    print confusion_matrix_all

    d_low_g_high = 0
    d_high_g_low = 0
    for tid in tweets:
        man_label = MAN_2_NUM[tweets[tid][GROUND_TRUTH_DIFF]]
        def_label = tweets[tid][DEF_DIFFICULTY]
        # G = high, D = low
        if man_label == "HD" and def_label == "LD":
            d_low_g_high += 1
        # G = low, D = high
        if man_label == "LD" and def_label == "HD":
            d_high_g_low += 1
    print "G=high, D=low:", d_low_g_high
    print "G=low, D=high", d_high_g_low


def simple_majority(tweets):
    """
    Compute worker disagreement according to simple majority voting (majority
    label has more votes than 2nd best label).

    Parameters
    ----------
    tweets: dict - dictionary of tweets from the dataset.

    Returns
    -------
    dict.
    {tid:
         {
            <TEXT>: tweet text,
            <MAJORITY_LABEL>: majority label,
            <DEF_DIFFICULTY>: HD or LD    # high or low disagreement
         }
    }
    """
    res = {}
    d = 0
    a = 0
    for tid in tweets:
        labels = tweets[tid][LABEL]

        # Consider only first k votes
        distrib = Counter(labels)
        majority, count1 = distrib.most_common()[0]
        disag = "LD"  # Low disagreement
        print "{} majority: {} count: {}".format(labels, majority, count1)
        # No unanimous decision
        if len(distrib.most_common()) > 1:
            second, count2 = distrib.most_common()[1]
            print "{} 2nd: {} count: {}".format(labels, second, count2)

            if count1 == count2:
                disag = "HD"
                d += 1
            else:
                a += 1
        else:
            a += 1
        print "disagreement:", disag
        res[tid] = {}
        res[tid][TEXT] = tweets[tid][TEXT]
        res[tid][MAJORITY_LABEL] = majority
        res[tid][DEF_DIFFICULTY] = disag
        res[tid][EXPLANATION] = ""
    print "#HD:", d
    print "#LD:", a
    return res


def read_tweets_from_csv(src, is_flat):
    """
    Reads in tweets from dataset in csv format separated by commas.

    Parameters
    ----------
    src: str - path where csv file is stored.
    is_flat: bool - True if <src> uses the flat labeling scheme. Otherwise it's
    assumed that the hierarchical one is used.

    Returns
    -------
    dict.
    Tweets.
    {tid:
        {
            <VOTES>: ...., # Number of workers who labeled the tweet
            # Annotation time of i-th annotator for j-th hierarchy level
            <TIMES>: [[time1 by anno1, time2 by anno1, time3 by anno1], [..]]
            <LABEL>: [[label1 by anno1, label2 by anno1, label3 by anno1], [..]]
            <UID>: .... # ID of tweet author (to retrieve tweets from Twitter)
            <WID>: [ID of anno1 who labeled this tweet, ID of anno2...]
            <TYPE>: ...
            <TEXT>: ... # Tweet message
            <GROUND_TRUTH_DIFF>: ... # Ground truth difficulty label, either
            (A)mbiguity, Lack of (B)ackground, (I)rrelevance, (O)ther, (E)asy
            (=(S)implicity in the paper)
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        tweets = {}
        for row in reader:
            tid, uid, ds_name, diff_label, text, votes, rest = \
                row[0], row[1], row[2], row[3], row[4], int(row[5]), row[6:]
            labels = []
            times = []
            if is_flat:
                wids, labelss, timess = rest[:votes], rest[votes: 2*votes], \
                                        rest[2*votes:]
                # 1 annotation time per worker
                for t in timess:
                    times.append(float(t))
                # 1 label per worker
                for l in labelss:
                    labels.append(l)
            else:
                wids, labelss, timess = rest[:votes], \
                    rest[votes: 3*votes+votes], rest[3*votes+votes:]

                # 3 labels per worker
                for i in xrange(0, len(labelss), 3):
                    one, two, three = labelss[i], labelss[i+1], labelss[i+2]
                    labels.append([one, two, three])
                # 3 annotation times per worker

                for i in xrange(0, len(timess), 3):
                    one, two, three = float(timess[i]), float(timess[i+1]), \
                                      float(timess[i+2])
                    times.append([one, two, three])
            # Store info for tweet
            tweets[tid] = {}
            tweets[tid][VOTES] = votes
            tweets[tid][LABEL] = labels
            tweets[tid][WID] = wids
            tweets[tid][UID] = uid
            tweets[tid][TIMES] = times
            tweets[tid][TYPE] = ds_name
            tweets[tid][TEXT] = text
            tweets[tid][GROUND_TRUTH_DIFF] = diff_label
    return tweets


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Read in manually labeled (w.r.t. difficulty) tweets that were labeled
    # using Excel
    src = os.path.join(base_dir, "results", "ambiguous_difficult",
                       "difficulty_labels_all_votes_correct_labeled.csv")

    # 1. Get derived (from majority voting) labels D and ground truth labels G
    # For G we have the labels:
    # (A)mbiguity
    # Lack of (B)ackground knowledge
    # (I)rrelevance
    # (O)ther
    # (E)asy (it's called "(S)implicity" in the paper)
    excel_tweets = read_excel_tweets(src)
    print "annotated in excel:", len(excel_tweets)

    # 2. Compute correlation between D and G
    # Not applicable because it's not continuous data, but categorical
    # compute_pearson_r(excel_tweets)
    compute_cramers_v(excel_tweets)

    # 3. Create confusion matrix as follows:
    #   * if D=low, G=high (=A, B, I, or O) -> TP (but we need to
    #     analyze why D=low)
    #   * if D=high, G=low (=E) -> TN (but we need to analyze why
    #     D=high, probably due to low-quality labels)
    create_confusion_matrix(excel_tweets)

    ###################################
    # Experiment: how would correlation look like with simple majority voting?
    ############################################
    # 4. Load tweets
    # Choose all tweets from TRAIN + LOW + MEDIUM + HIGH
    src = os.path.join(base_dir, "results", "export",
                       "combined_tweets.csv")
    tweets = read_tweets_from_csv(src, True)
    # 5. Derive disagreement labels D
    new_tweets = simple_majority(tweets)
    # Overwrite D with newly derived D
    for tid in excel_tweets:
        excel_tweets[tid][DEF_DIFFICULTY] = new_tweets[tid][DEF_DIFFICULTY]
    # 6. Compute correlation between D and G
    compute_cramers_v(excel_tweets)
    # 7. Create confusion matrix
    create_confusion_matrix(excel_tweets)
    # Result
    # disagreement_diff_label  HD    LD
    # man_diff_label
    # A                        82   225
    # B                        77   256
    # E                        94  2332
    # I                        86   219
    # O                        30    99
    # 0.328733147171
    # disagreement_diff_label   HD    LD
    # man_diff_label
    # HD                       275   799
    # LD                        94  2332
    # G=high, D=low: 799
    # G=low, D=high 94
