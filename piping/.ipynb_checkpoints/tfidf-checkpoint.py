'''make vectorizer from sequences'''
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
import pandas as pd  # type: ignore
import pickle
import numpy as np  # type: ignore
import argparse
from subprocess import call

parser = argparse.ArgumentParser(description="give hyperparameters to TfidfVectorizer and create a pickled model")
parser.add_argument(
    "--min_f_count",
    default=16,
    type=int,
    help="min_f_count: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.")
parser.add_argument(
    "--ngram_max",
    default=8,
    type=int,
    help="ngram_max: max length of substring for tokenizer")
parser.add_argument(
    "--max_features",
    default=10000,
    type=int,
    help="throttle the maximum number of resulting features")
parser.add_argument(
    "--downsample",
    default=-1,
    type=int,
    help="downsample: throttle from read_csv, process less data")

args = parser.parse_args()

raw_prefix = 'https://raw.githubusercontent.com/prescriptive-possibilities-april-15-19/mocking/master/'


def seq_vectorizer(
        min_f_count: int = 10,
        ngram_max: int = 10,
        max_features: int = 10000,
        downsample: int = -1
        ) -> TfidfVectorizer:
    if downsample < 16:
        sequences = pd.read_csv(
            raw_prefix + 'sequences.csv').rename({'Unnamed: 0': 'seq_id'}, axis=1)
    else:
        sequences = pd.read_csv(raw_prefix + 'sequences.csv').rename(
            {'Unnamed: 0': 'seq_id'}, axis=1).sample(downsample)

    tfidf = TfidfVectorizer(
        lowercase=False,
        analyzer='char',
        stop_words=None,
        ngram_range=(1, ngram_max),
        min_df=min_f_count,
        max_features=max_features
    )

    print("training data: ", sequences.shape)

    print("fitting........") 
    tfidf.fit(sequences.sequence.values)
    print("all trained up!")
    return tfidf


if __name__ == '__main__':
    print("will fit sequences.csv to TF-IDF with parameters", args)
    sequences_tfidf_model = seq_vectorizer(
        min_f_count=args.min_f_count,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
        downsample=args.downsample)

    print("training done!")

    print("pickling.....")
    with open("tfidf.pickle", "wb") as fp:
        pickle.dump(sequences_tfidf_model, fp)
    print("your pickle is size ", call("wc -c tfidf.pickle", shell=True))

    
