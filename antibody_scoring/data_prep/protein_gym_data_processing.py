"""Contains code needed to preprocess the ProteinGym data."""
import os
from math import ceil
import pandas as pd
import numpy as np



def extract_tt_splits(random_seed, sequence_list, ydata_list,
        method = "random"):
    """Returns a train-test split using a specified method."""
    if method == "random":
        rng = np.random.default_rng(random_seed)
        idx = rng.permutation(len(sequence_list)).tolist()

        sequence_batches, y_batches = [], []
        batch_size = ceil(len(idx) / 5)

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            sequence_batches.append( [sequence_list[j] for j in batch_idx] )
            y_batches.append( [ydata_list[j] for j in batch_idx] )

        return sequence_batches, y_batches

    raise RuntimeError("Other split methods aside from random not "
                "yet supported.")




def preprocess_dms_substitutions(project_dir):
    """Preps the data associated with dms substitutions."""
    current_dir = os.getcwd()
    os.chdir(os.path.join(project_dir, "extracted_data", "protein_gym_dms_subs"))

    flist = [os.path.abspath(f) for f in os.listdir() if f.endswith(".csv")]
    os.chdir(current_dir)

    for fname in flist:
        raw_data = pd.read_csv(fname)
        yield raw_data["mutated_sequence"].tolist(), \
                raw_data["DMS_score"].tolist(), fname



def preprocess_dms_indels(project_dir):
    """Preps the data associated with dms indels."""
    current_dir = os.getcwd()
    os.chdir(os.path.join(project_dir, "extracted_data", "protein_gym_dms_indels"))

    flist = [os.path.abspath(f) for f in os.listdir() if f.endswith(".csv")]
    os.chdir(current_dir)

    for fname in flist:
        raw_data = pd.read_csv(fname)
        yield raw_data["mutated_sequence"].tolist(), \
                raw_data["DMS_score"].tolist(), fname
