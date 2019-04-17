'''make a models dictionary

each model is a trained BaggingClassifierPU, https://roywright.me/2017/11/16/positive-unlabeled-learning/
'''
# from pandarallel import pandarallel
# pandarallel.initialize()
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
import pandas as pd  # type: ignore
import pickle
import numpy as np  # type: ignore
import argparse
from subprocess import call
from typing import Tuple, List
from functools import reduce
import sys
from utils import Spinner

from baggingPU import BaggingClassifierPU
from tfidf import seq_vectorizer, raw_prefix

raw_prefix = 'https://raw.githubusercontent.com/prescriptive-possibilities-april-15-19/mocking/master/'

seed = 42
np.random.seed(seed)
lig_num = 1000
sample_size = 200
estimators = 100
s_ = sample_size//2 - 2

downsample_binding=230000
downsample_sequences=20000

def init_dat(lig_num: int = lig_num,
             sample_size: int = sample_size,
             estimators: int = estimators,
             downsample_binding: int = -1,
             downsample_sequences: int = -1) -> Tuple[pd.DataFrame, List[int], pd.DataFrame]:

    def lig2seq_():
        return (pd.read_csv(raw_prefix+'lig2seq.csv')
                .rename({'lig': 'lig_idx','seq': 'seq_idx'}, axis=1))

    def sequences_():
        return pd.read_csv(raw_prefix+'sequences.csv',
                           index_col=0
                          )

    if downsample_binding < 16:
        lig2seq = lig2seq_()
    else:
        lig2seq = lig2seq_().sample(downsample_binding)

    if downsample_sequences < 16:
        sequences = sequences_()
    else:
        sequences = sequences_.sample(downsample_sequences)

    lig_id_vals = list(np.random.choice(lig2seq.lig_idx.unique(), size=lig_num))
    binding = lig2seq.loc[lig2seq.lig_idx.isin(lig_id_vals)]

    return sequences, lig_id_vals, binding

def fitter_df_maker(lig_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def labeled_(psqs):
        labeled_seqs = sequences.loc[sequences.index.isin(psqs)] # hidden_0

        labeled_seqs_known = labeled_seqs.sample(frac=0.75)

        labeled_seqs_hidden = labeled_seqs.loc[~labeled_seqs.index.isin(labeled_seqs_known.index)]
        return labeled_seqs_known, labeled_seqs_hidden

    positive_seq_ids = binding.loc[binding.lig_idx==lig_id, 'seq_idx'].values

    if len(positive_seq_ids) > s_: # we want unlabeled to be dominant
        positive_seq_ids = np.choice(positive_seq_ids, s_)
    if len(positive_seq_ids) > 5:
        unlabeled_seqs = (sequences.loc[~sequences.index.isin(positive_seq_ids)]
                          .sample(n = sample_size-len(positive_seq_ids))) #

        labeled_seqs_known, labeled_seqs_hidden = labeled_(positive_seq_ids)

        unlabeled_seqs['bind'] = np.zeros(unlabeled_seqs.shape[0]) # equiv to df_seq_sub_neg.loc[:,"bind"] = 0

        labeled_seqs_known['bind'] = np.ones(labeled_seqs_known.shape[0])

        labeled_seqs_hidden['bind'] = np.zeros(labeled_seqs_hidden.shape[0])

        df_fitter = pd.concat([unlabeled_seqs, labeled_seqs_known, labeled_seqs_hidden])

        X = pd.DataFrame(tfidf.transform(df_fitter.sequence.values).toarray(),
                         columns=tfidf.get_feature_names(),
                         index=df_fitter.index)

        y = df_fitter.bind
        #print(lig_id, X.shape, y.shape)
        return X,y
    else:
        raise Exception


sequences, lig_id_vals, binding = init_dat(
    downsample_binding=downsample_binding,
    #downsample_sequences=downsample_sequences # you really can't do this, if you downsample here you'll throw off the ligand/sequence correspondence.
)



tfidf = seq_vectorizer(ngram_max=4, downsample=40000)

print(f"FOR BaggingClassifierPU MODELS:\t Sequences: {sequences.shape};\t Binding: {binding.shape};\t Number of ligand id values: {len(lig_id_vals)}. ")

spinner = Spinner()
models = {}
for lig_id in lig_id_vals:
    try:
        X, y = fitter_df_maker(lig_id)
        sys.stdout.write(f"populating models dict now at length {len(models.keys())}... ")
        bc = BaggingClassifierPU(DecisionTreeClassifier(),
                                 n_estimators=estimators,
                                 #n_jobs=-1,
                                 max_samples=int(sum(y.values)))

        sys.stdout.write(f"next, fitting on ligand #{lig_id}.")
        spinner.start
        bc.fit(X,y)
        spinner.stop
        models[lig_id] = bc
        #sys.stdout.flush()
        sys.stdout.write('\r') # yes finally https://stackoverflow.com/questions/23138413/clearing-old-data-from-sys-stdout-in-python
    except:
        pass

#with open('models.pickle', 'wb') as mp:
#    pickle.dump(models, mp)
