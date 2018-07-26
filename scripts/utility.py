"""
Contains functions that might be used across different scripts
"""
from pymongo import MongoClient

# Indicates that a tweet has no labels
NA = " "

# Valid DB names:
# lturannotationtool: L run Sabanci
# mdlannotationtool: L run MD (Michael)
# mdsmannotationtool: S/M run MD
# turannotationtool: S/M run Sabanci
# vishnuannotationtool: Vishnu annotated additionally 200 tweets to his 150 here
# harishannotationtool: Harish annotated additionally 50 tweets to his 150 here. done after election.
# kantharajuannotationtool: Kantharaju annotated additionally 50 tweets to his. done after election.
# mdslaterannotationtool: volunteers performing experiment in Jan. done after election.
# in MD
def load_tweets_annotators_from_db(db_name, tweet_coll_name="tweets",
                              user_coll_name="user"):
    """
    Loads tweets and users from a given MongoDB assuming they exist.

    Parameters
    ----------
    db_name: str - name of the MongoDB to be opened.
    tweet_coll_name: str - name of the collection holding the tweets.
    user_coll_name: str - name of the collection holding the users.

    Returns
    -------
    pymongo.collection.Collection, pymongo.collection.Collection
    Collection of tweets, collection of annotators. You can access the values by
    specifiying the keys (=field names in collection).

    """
    connection = MongoClient()
    # print "The following MongoDBs exist: {}".format(connection.database_names())
    # print "DB to be opened: {}".format(db_name)
    # Our database
    db = connection[db_name]
    users = db[user_coll_name]
    tweets = db[tweet_coll_name]
    return tweets, users


def get_annotated_tweets_by_user(user_collection, username):
    """
    Finds the tweet IDs of the annotated tweets by a certain annotator.

    Parameters
    ----------
    user_collection: pymongo.collection.Collection - collection in which users
    are stored.
    username: string - name of the user for which the annotated tweets should
    be found.

    Returns
    -------
    List of ObjectIds.
    Each ID represents a tweet ID.

    """
    user = user_collection.find_one({"username": username})
    return user["annotated_tweets"]


def get_tweet(tweet_collection, t_id):
    """
    Returns a tweet from the collection.

    Parameters
    ----------
    tweet_collection: pymongo.collection.Collection - tweet collection.
    t_id: bson.ObjectId - ID of the tweet.

    Returns
    -------
    dict.
    Dictionary containing all values from the DB. Keys are the same ones as in
    the DB.

    """
    tweet = tweet_collection.find_one({"_id": t_id})
    return tweet


def get_tweet_by_twitter_id(tweet_collection, t_id):
    """
    Returns a tweet from the collection.

    Parameters
    ----------
    tweet_collection: pymongo.collection.Collection - tweet collection.
    t_id: str - Twitter ID of the tweet.

    Returns
    -------
    dict.
    Dictionary containing all values from the DB. Keys are the same ones as in
    the DB.

    """
    tweet = tweet_collection.find_one({"id_str": t_id})
    return tweet


if __name__ == "__main__":
    tweets, annotators = load_tweets_annotators_from_db("lturannotationtool")
    for anno in annotators:
        print anno["username"]
