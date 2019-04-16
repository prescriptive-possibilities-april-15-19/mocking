"""partially supervised learning"""

from tfidf import seq_vectorizer, raw_prefix
from baggingPU import BaggingClassifierPU
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pickle

seed = 42
np.random.seed(seed)
lig_num = 1000
sample_size = 200
estimators = 100

downsample_binding = 230000
downsample_sequences = 20000


def init_dat(lig_num: int = lig_num, sample_size: int = sample_size,
             estimators: int = estimators, downsample_binding: int = -1, downsample_sequences: int = -1):

    if downsample_binding < 16:
        lig2seq = pd.read_csv(
            raw_prefix + 'lig2seq.csv').rename({'lig': 'lig_id', 'seq': 'seq_id'}, axis=1)
    else:
        lig2seq = pd.read_csv(
            raw_prefix +
            'lig2seq.csv').rename(
            {
                'lig': 'lig_id',
                'seq': 'seq_id'},
            axis=1).sample(downsample_binding)

    if downsample_sequences < 16:
        sequences = pd.read_csv(
            raw_prefix + 'sequences.csv').rename({'Unnamed: 0': 'seq_id'}, axis=1)
    else:
        sequences = pd.read_csv(raw_prefix + 'sequences.csv').rename(
            {'Unnamed: 0': 'seq_id'}, axis=1).sample(downsample_sequences)

    lig_id_vals = list(np.random.choice(lig2seq.lig_id.unique(), size=lig_num))
    binding = lig2seq.loc[lig2seq.lig_id.isin(lig_id_vals)]

    return sequences, lig_id_vals, binding


sequences, lig_id_vals, binding = init_dat(
    downsample_binding=downsample_binding, downsample_sequences=downsample_sequences)

# continue with the rest of this code https://github.com/prescriptive-possibilities-april-15-19/DS-data/blob/development-michael/notebooks/PUC_protein_ligand.ipynb
# ...
