"""partially supervised learning"""

from tfidf import seq_vectorizer, raw_prefix
raw_prefix = 'https://raw.githubusercontent.com/prescriptive-possibilities-april-15-19/mocking/master/'
from baggingPU import BaggingClassifierPU  # ignore: type
from sklearn.tree import DecisionTreeClassifier  # ignore: type
import pandas as pd  # ignore: type
import numpy as np # ignore: type
import pickle
from typing import Tuple, List

seed = 42
np.random.seed(seed)
lig_num = 1000
estimators = 100

downsample_binding = 230000
downsample_sequences = 20000

sample_size = 800
s_ = np.floor(sample_size / 2 - sample_size / 100)

sequences_tfidf_model = seq_vectorizer(ngram_max=3)
#with open('tfidf.pickle', 'rb') as p: 
#    sequences_tfidf_model = p


def init_dat(lig_num: int = lig_num, sample_size: int = sample_size,
             estimators: int = estimators, downsample_binding: int = -1,
             downsample_sequences: int = -1):

    def lig2seq_():
        return (pd.read_csv(raw_prefix + 'lig2seq.csv')
                .rename({'lig': 'lig_idx', 'seq': 'seq_idx'}, axis=1))

    def sequences_():
        return pd.read_csv(raw_prefix + 'sequences.csv',
                           index_col=0
                           )

    if downsample_binding < 16:
        lig2seq = lig2seq_()
    else:
        lig2seq = lig2seq_().sample(downsample_binding)

    if downsample_sequences < 16:
        sequences = sequences_()
    else:
        sequences = sequences_().sample(downsample_sequences)

    lig_id_vals = list(
        np.random.choice(
            lig2seq.lig_idx.unique(),
            size=lig_num))
    binding = lig2seq.loc[lig2seq.lig_idx.isin(lig_id_vals)]

    return sequences, lig_id_vals, binding


sequences, lig_id_vals, binding = init_dat(
    downsample_binding=downsample_binding,
    # downsample_sequences=downsample_sequences
)

print(sequences.shape, binding.shape, len(lig_id_vals))


def fitter_df_maker(lig_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def labeled_(psqs):
        labeled_seqs = sequences.loc[sequences.index.isin(psqs)]  # hidden_0

        labeled_seqs_known = labeled_seqs.sample(frac=0.75)

        labeled_seqs_hidden = labeled_seqs.loc[~labeled_seqs.index.isin(
            labeled_seqs_known.index)]
        return labeled_seqs_known, labeled_seqs_hidden

    positive_seq_ids = binding.loc[binding.lig_idx == lig_id, 'seq_idx'].values
    if len(positive_seq_ids) > s_:  # we want unlabeled to be dominant
        positive_seq_ids = np.random.choice(positive_seq_ids, s_)
    if len(positive_seq_ids) > 5:
        unlabeled_seqs = (sequences.loc[~sequences.index.isin(positive_seq_ids)]
                          .sample(n=sample_size - len(positive_seq_ids)))

        labeled_seqs_known, labeled_seqs_hidden = labeled_(positive_seq_ids)

        # equiv to df_seq_sub_neg.loc[:,"bind"] = 0
        unlabeled_seqs['bind'] = np.zeros(unlabeled_seqs.shape[0])

        labeled_seqs_known['bind'] = np.ones(labeled_seqs_known.shape[0])

        labeled_seqs_hidden['bind'] = np.zeros(labeled_seqs_hidden.shape[0])

        df_fitter = pd.concat(
            [unlabeled_seqs, labeled_seqs_known, labeled_seqs_hidden])

        X = pd.DataFrame(
            sequences_tfidf_model.transform(
                df_fitter.sequence.values).toarray(),
            columns=sequences_tfidf_model.get_feature_names(),
            index=df_fitter.index)

        y = df_fitter.bind
        print(X.shape, y.shape)
        return X, y
    else:
        raise Exception
