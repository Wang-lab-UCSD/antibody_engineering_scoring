"""Contains functionality for preprocessing and encoding the datasets."""
import os
import pandas as pd
import numpy as np
from antpack import SingleChainAnnotator as SCA
from .constants import data_retrieval_constants as DRC


def preprocess_engelhart(project_dir):
    """Preps the Engelhart data and returns both 1) a list of the CDRs
    or 2) the numbered CDRs concatenated so that the result is a fixed-length
    sequence (this is more convenient for XGBoost)."""
    os.chdir(os.path.join(project_dir, "extracted_data", "engelhart", DRC.LOCAL_ENGELHART_PATH[0],
                        DRC.LOCAL_ENGELHART_PATH[1]))
    raw_data = pd.read_csv("MITLL_AAlphaBio_Ab_Binding_dataset.csv")
    os.chdir(project_dir)

    raw_data = raw_data[raw_data["Target"]=="MIT_Target"]

    # As in Barton et al., if there are multiple values for one sequence, average over them.
    # Also, eliminate any sequences for which the label is in the range 3 - 4 (as in Barton et al.)
    agg_fun = {col:"first" for col in raw_data.columns.tolist()}
    agg_fun["Pred_affinity"] = "mean"
    raw_data = raw_data.groupby('Sequence').aggregate(agg_fun)

    raw_data = raw_data.loc[(raw_data['Pred_affinity'] < 3) | (raw_data['Pred_affinity'] >= 4)].copy()
    cdr_list = ["CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"]

    # For xGPR convolution kernels, the sequences do not have to be the same length, so we can just
    # input them -- this is straightforward. We just need to get the sequence length.
    cdr_unaligned = [raw_data[col].tolist() for col in cdr_list]
    slengths = [np.array([len(s) for s in cdrset]).astype(np.int32) for cdrset in cdr_unaligned]



    # For XGBoost, however, the sequences must all be aligned. We could just number using AntPack and extract
    # CDRs, but we need to make sure the CDRs we extract match the CDRs in the dataframe, so we adopt a
    # more complicated procedure. We start by generating AntPack numbering.
    sca_heavy = SCA(chains=["H"])
    sca_light = SCA(chains=["L"])
    cdr_numbering = [[] for cdr in cdr_list]

    for i, seq in enumerate(raw_data["Sequence"].tolist()):
        h_res = sca_heavy.analyze_seq(seq)[0]
        for j, cdr in enumerate(cdr_list[:3]):
            cdr_seq = raw_data[cdr].values[i]
            begin = seq.find(cdr_seq)
            end = begin + len(cdr_seq)
            cdr_numbering[j].append(h_res[begin:end])

        l_res = sca_light.analyze_seq(seq)[0]
        for j, cdr in enumerate(cdr_list[3:]):
            cdr_seq = raw_data[cdr].values[i]
            begin = seq.find(cdr_seq)
            end = begin + len(cdr_seq)
            cdr_numbering[j+3].append(l_res[begin:end])


    # Ordinarily some positions would contain letters -- in this case however
    # none do, so we can just sort. (The order doesn't actually matter as long
    # as say IMGT position 36 is placed in the same position in the output.)
    expected_positions = {}

    for i, cdr in enumerate(cdr_list):
        unique_numbers = []
        for c in cdr_numbering[i]:
            unique_numbers += c

        expected_positions[cdr] = np.sort(np.unique(unique_numbers))


    full_length = sum([len(d) for (_, d) in expected_positions.items()])

    net_length = 0
    encoded_seqs = [["-" for k in range(full_length)] for s in raw_data["Sequence"].tolist()]

    for j, cdr in enumerate(cdr_list):
        seqs = raw_data[cdr].tolist()
        seq_idx = {c:(i+net_length) for i, c in enumerate(expected_positions[cdr])}

        for seq, encoded_seq,numbering in zip(seqs, encoded_seqs, cdr_numbering[j]):
            for letter, pos in zip(seq, numbering):
                encoded_seq[seq_idx[pos]] = letter

        net_length += len(seq_idx)

    encoded_seqs = ["".join(s) for s in encoded_seqs]

    yvalues = raw_data["Pred_affinity"].values.astype(np.float64)
    del raw_data
    return encoded_seqs, yvalues, cdr_unaligned, slengths
