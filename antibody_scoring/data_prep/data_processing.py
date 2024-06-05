"""Contains functionality for preprocessing and encoding the datasets."""
import os
import pandas as pd
import numpy as np
from antpack import SingleChainAnnotator as SCA
from antpack import VJGeneTool as VJG
from sklearn.model_selection import train_test_split
from ..constants import data_retrieval_constants as DRC
from ..constants.seq_encoding_constants import DESAUTELS_VARIABLE_POSITIONS
from ..constants.seq_encoding_constants import IL6_ANTIGEN_VARIABLE_POSITIONS


def preprocess_engelhart(project_dir, regression_only = False):
    """Preps the Engelhart data and returns both 1) a list of the CDRs
    or 2) the numbered CDRs concatenated so that the result is a fixed-length
    sequence (this is more convenient for XGBoost)."""
    os.chdir(os.path.join(project_dir, "extracted_data", "engelhart", DRC.LOCAL_ENGELHART_PATH[0],
                        DRC.LOCAL_ENGELHART_PATH[1]))
    raw_data = pd.read_csv("MITLL_AAlphaBio_Ab_Binding_dataset.csv")
    os.chdir(project_dir)

    raw_data = raw_data[raw_data["Target"]=="MIT_Target"]
    raw_data = raw_data.iloc[~np.isnan(raw_data["Pred_affinity"].values),:]

    # As in Barton et al., if there are multiple values for one sequence, average over them.
    agg_fun = {col:"first" for col in raw_data.columns.tolist()}
    agg_fun["Pred_affinity"] = "mean"
    raw_data = raw_data.groupby('Sequence').aggregate(agg_fun)

    # Also, eliminate any sequences for which the label is in the range 3 - 4 (as in Barton et al.),
    # UNLESS we are instead performing a comparison to Deutschmann et al. in which case we should not
    # do this.
    if not regression_only:
        raw_data = raw_data.loc[(raw_data['Pred_affinity'] < 3) |
                (raw_data['Pred_affinity'] >= 4)].copy()
    cdr_list = ["CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"]

    # For xGPR convolution kernels, the sequences do not have to be the same length, so we can just
    # input them -- this is straightforward. We just need to get the sequence length.
    cdr_unaligned = [raw_data[col].tolist() for col in cdr_list]
    cdr_slengths = [np.array([len(s) for s in cdrset]).astype(np.int32) for cdrset in cdr_unaligned]



    # For XGBoost, however, the sequences must all be aligned. We could just number
    # using AntPack and extract CDRs, but we need to make sure the CDRs we extract
    # match the CDRs in the dataframe, so we adopt a more complicated procedure.
    # We start by generating AntPack numbering.
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


    expected_positions = {}

    for i, cdr in enumerate(cdr_list):
        unique_numbers = set()
        for c in cdr_numbering[i]:
            unique_numbers.update(c)

        unique_numbers = [u for u in list(unique_numbers) if u != "-"]
        expected_positions[cdr] = sca_heavy.sort_position_codes(unique_numbers,
                                    scheme = "imgt")


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

    seq_unaligned = raw_data["Sequence"].tolist()
    seq_slengths = [len(s) for s in seq_unaligned]

    return encoded_seqs, yvalues, cdr_unaligned, cdr_slengths, \
            seq_unaligned, seq_slengths


def preprocess_mason(project_dir, fraction = 0.49506):
    """Preprocesses the Mason et al. dataset for evaluation. Most
    of the processing code here is directly copied from the DMS_opt retrieval,
    although we have had to make some modifications to more closely mirror
    the preprocessing from Barton et al. In particular, we have made some
    adjustments to ensure the training / test data sizes match Barton et al. -- 
    in particular, the value chosen for fraction, also the procedure for
    dropping duplicate sequences."""
    Ag_pos = pd.read_csv(os.path.join(project_dir, "extracted_data", "mason",
        "mHER_H3_AgPos.csv"))
    Ag_neg = pd.read_csv(os.path.join(project_dir, "extracted_data", "mason",
        "mHER_H3_AgNeg.csv"))

    class Collection:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

    # The original DMS_opt repo drops duplicates AFTER merging the dfs, but this
    # results in a different number of train and test dpoints than were used by
    # Barton et al, who therefore presumably filtered first (although their code
    # was not provided). We therefore use that approach here.
    Ag_combined = pd.concat([Ag_pos, Ag_neg])
    Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_pos = Ag_combined[Ag_combined["AgClass"]==1].copy().reset_index()
    Ag_neg = Ag_combined[Ag_combined["AgClass"]==0].copy().reset_index()
    del Ag_combined

    # Calculate data sizes based on ratio
    data_size_pos = len(Ag_pos)/fraction
    data_size_neg = len(Ag_neg)/(1-fraction)

    # Adjust the length of the data frames to meet the ratio requirement
    if len(Ag_pos) <= len(Ag_neg):
        if data_size_neg < data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*fraction))]
            Ag_neg1 = Ag_neg
            Unused = Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)]

        if data_size_neg >= data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_pos*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_pos*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_pos*(1-fraction))):len(Ag_neg)]]
            )
    else:
        if data_size_pos < data_size_neg:
            Ag_pos1 = Ag_pos
            Ag_neg1 = Ag_neg[0:(int(data_size_pos*(1-fraction)))]
            Unused = Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)]

        if data_size_pos >= data_size_neg:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_neg*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_neg*(1-fraction))):len(Ag_neg)]]
            )
    # Combine the positive and negative data frames
    # Original function did not supply a random seed here, fixed
    Ag_combined = pd.concat([Ag_pos1, Ag_neg1])
    #Ag_combined = Ag_combined.drop_duplicates(subset='AASeq')
    Ag_combined = Ag_combined.sample(frac=1, random_state=0).reset_index(drop=True)

    # 80%/20% training test data split
    # Original function did not supply a random seed here, fixed
    idx = np.arange(0, Ag_combined.shape[0])
    idx_train, idx_test = train_test_split(
        idx, stratify=Ag_combined['AgClass'], test_size=0.2,
        random_state=0
    )
    
    # 50%/50% test validation data split
    # Original function did not supply a random seed here, fixed
    idx2 = np.arange(0, idx_test.shape[0])
    idx_val, idx_test2 = train_test_split(
        idx2, stratify=Ag_combined.iloc[idx_test, :]['AgClass'], test_size=0.5,
        random_state=0
    )

    # Create collection
    Seq_Ag_data = Collection(
        train=Ag_combined.iloc[idx_train, :],
        val=Ag_combined.iloc[idx_test, :].iloc[idx_val, :],
        test=Ag_combined.iloc[idx_test, :].iloc[idx_test2, :],
        complete=Ag_combined
    )
    #Add unused to training set and shuffle
    #Seq_Ag_data.train = pd.concat([Seq_Ag_data.train, Unused])
    #Seq_Ag_data.train = Seq_Ag_data.train.sample(frac=1, random_state=0)

    # Here also we have modified the original function. Rather than returning
    # all of the data, we just return _all_ sequences (training, validation and
    # test). This enables caller to select the training / test set itself without
    # any further modifications to this code.
    return Seq_Ag_data.complete["AASeq"].tolist(), Seq_Ag_data.complete["AgClass"].values



def preprocess_desautels(project_dir):
    """Preps the data for Desautels et al."""
    raw_data = pd.read_csv(os.path.join(project_dir, "extracted_data", "desautels",
        "biorxiv_paper", "Desautels_insilico_data.csv"))

    # These are the same pred cols used by Singh et al.
    pred_cols = ['FoldX_Average_Whole_Model_DDG', 'FoldX_Average_Interface_Only_DDG',
            'Statium', 'Sum_of_Rosetta_Flex_single_point_mutations',
            'Sum_of_Rosetta_Total_Energy_single_point_mutations']

    retained_cols = ["Antibody_Sequence"] + pred_cols
    raw_data = raw_data[retained_cols].copy()
    raw_data = raw_data.groupby('Antibody_Sequence').mean().reset_index()

    yarr = raw_data[pred_cols].values.astype(np.float64)
    unprepped_seqs = raw_data["Antibody_Sequence"].tolist()

    # There are only certain positions in these antibodies which vary. We COULD feed the whole
    # sequence into a convolution kernel, but it's much faster to just use the variable positions.
    seqs = ["".join([unprepped_seq[i] for i in DESAUTELS_VARIABLE_POSITIONS])
                for unprepped_seq in unprepped_seqs]

    return seqs, yarr


def preprocess_cognano(project_dir):
    """Preprocess the Cognano dataset."""
    test_data = pd.read_csv(os.path.join(project_dir, "extracted_data",
        "cognano", "test.csv"))
    train_data = pd.read_csv(os.path.join(project_dir, "extracted_data",
        "cognano", "train.csv"))

    datagroups = [["train", train_data], ["test", test_data]]
    sca = SCA(chains=["H"], scheme="imgt")

    observed_positions, observed_antigens = set(), set()

    for i, datagroup in enumerate(datagroups):
        numberings = []
        for seq, ag in zip(datagroup[1]["VHH_sequence"].tolist(),
                datagroup[1]["Ag_label"].tolist()):
            numbering, _, _, _ = sca.analyze_seq(seq)
            observed_positions.update(numbering)
            observed_antigens.add(ag)
            numberings.append(numbering)

        datagroups[i].append(numberings)

    sorted_codes = sca.sort_position_codes([k for k in list(observed_positions) if
        k != "-"], scheme="imgt")
    sorted_codes = {k:i for i, k in enumerate(sorted_codes)}

    sorted_ags = sorted(list(observed_antigens))
    sorted_ags = {k:i for i, k in enumerate(sorted_ags)}

    prepped_sets = []

    for i, datagroup in enumerate(datagroups):
        prepped_seqs, prepped_antigens = [], []
        for seq, numbering, antigen in zip(datagroup[1]["VHH_sequence"].tolist(),
                datagroup[2], datagroup[1]["Ag_label"].tolist()):
            prepped_seq = ["-" for k in sorted_codes]
            for n, letter in zip(numbering, seq):
                if n == "-":
                    continue
                prepped_seq[sorted_codes[n]] = letter
            prepped_seqs.append("".join(prepped_seq))
            ag_encoded = np.zeros((len(sorted_ags)), dtype=np.uint8)
            ag_encoded[sorted_ags[antigen]] = 1
            prepped_antigens.append(ag_encoded)

        prepped_sets.append(  (prepped_seqs, np.stack(prepped_antigens),
            datagroup[1]["label"].tolist(), datagroup[0])  )

    return prepped_sets


def preprocess_il6(project_dir):
    """Preps the IL6 data for analysis. Barton et al. used a strangely complicated
    approach to preparing the data, including discarding any sequences with low
    percent identity to a human V-gene (?). We follow their procedure here for ease
    of comparison, although their procedure is probably not the best or most
    straightforward way to process this dataset."""
    raw_data = pd.read_csv(os.path.join(project_dir, "extracted_data",
        "il6", "il6_aai_dataset.csv"))

    sca = SCA(chains=["H"], scheme="imgt")

    observed_positions, numberings = set(), []

    numberings = []
    for seq in raw_data["VHH_sequence"].tolist():
        numbering, _, _, _ = sca.analyze_seq(seq)
        observed_positions.update(numbering)
        numberings.append(numbering)

    sorted_codes = sca.sort_position_codes([k for k in list(observed_positions) if
        k != "-"], scheme="imgt")
    sorted_codes = {k:i for i, k in enumerate(sorted_codes)}

    sorted_ag_labels = np.sort(np.unique(raw_data["Ag_label"].values))
    sorted_ag_labels = {k:i for i, k in enumerate(sorted_ag_labels.tolist())}

    unique_ag_seqs = {}
    for ag_label, ag_seq in zip(raw_data["Ag_label"].tolist(), raw_data["Ag_sequence"].tolist()):
        if ag_label not in unique_ag_seqs:
            prepped_ag = "".join([ag_seq[k] for k in IL6_ANTIGEN_VARIABLE_POSITIONS])
            unique_ag_seqs[ag_label] = prepped_ag

    prepped_seqs = {}
    vj_tool = VJG()

    for seq, numbering, label, ag_label in zip(raw_data["VHH_sequence"].tolist(),
                numberings, raw_data["label"].tolist(), raw_data["Ag_label"].tolist()):

        _, _, vident, _ = vj_tool.assign_numbered_sequence(seq, numbering, "H")
        if vident < 0.75:
            continue

        prepped_seq = ["-" for k in sorted_codes]
        for n, letter in zip(numbering, seq):
            if n == "-":
                continue
            prepped_seq[sorted_codes[n]] = letter

        prepped_seq = "".join(prepped_seq)

        if prepped_seq not in prepped_seqs:
            prepped_seqs[prepped_seq] = np.zeros((len(sorted_ag_labels)), dtype=np.uint8)

        if label == 1:
            prepped_seqs[prepped_seq][sorted_ag_labels[ag_label]] = 1
        elif label == 0:
            prepped_seqs[prepped_seq][sorted_ag_labels[ag_label]] = 2

    del raw_data

    # The next part of the filtering process used by Barton et al. is quite
    # strange -- they retain only sequences that bind one variant or no
    # variants, but remove sequences that bind one variant and are confirmed
    # not to bind any others (?). We duplicate this procedure here although it
    # is not clear to us why this was adopted.
    filtered_seqs = {}

    for prepped_seq, allocation_arr in prepped_seqs.items():
        n_binders = (allocation_arr == 1).sum()
        if n_binders > 1:
            continue

        confirmed_nonbinders = (allocation_arr == 2).sum()
        filtered_arr = allocation_arr.copy()
        filtered_arr[filtered_arr==2] = 0

        if confirmed_nonbinders == (allocation_arr.shape[0] - 1) and \
                n_binders == 1:
            filtered_arr[:] = 0

        filtered_seqs[prepped_seq] = filtered_arr

    return filtered_seqs
