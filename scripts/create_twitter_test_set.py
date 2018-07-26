"""
Creates a Twitter test set from the presidential election.
"""
import os
import codecs
import json
import unicodecsv as csv


if __name__ == "__main__":
    TWITTER = ""
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    DST = os.path.join(base_dir, "results", "dataset_twitter_full",
                       "tweets.csv")
    # Directory in which tweets from the dataset are stored
    FINAL = "/media/data/dataset/debate1_trump_vs_clinton_final_dataset/"
    # Directory in which all acceptable tweets without URLs are stored
    TWEETS = "/media/data/dataset/debate1_trump_vs_clinton_sanitized"

    # Get tweets that can't be used
    # {tid: None}
    ignored = {}
    available = {}
    for fname in os.listdir(FINAL):
        tid = fname.split("_")[1].split(".")[0]
        ignored[tid] = 0

    # Get remaining unlabeled tweets
    for fname in os.listdir(TWEETS):
        tid = fname.split("_")[1].split(".")[0]
        if tid not in ignored:
            p = os.path.join(TWEETS, fname)
            with codecs.open(p, "rb", encoding="utf-8") as f:
                tweet = json.load(f, encoding="utf-8")
            text = tweet["text"]
            # Remove all line breaks in a tweet
            text = text.replace('\n', ' ').replace('\r', '')
            available[tid] = text
            if "\t" in available[tid]:
                print "tab", tid

    # Store in csv file
    with open(DST, "wb") as f:
        writer = csv.writer(f, dialect='excel', encoding='utf-8',
                            delimiter="\t")
        # Write header
        for tid in available:
            text = available[tid]
            writer.writerow([tid, text])
