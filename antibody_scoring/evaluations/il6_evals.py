"""Contains code for evaluating xgboost on the IL-6 dataset."""
import time
import random
import optuna
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
import numpy as np
from ..data_prep.data_processing import preprocess_il6
from ..data_prep.seq_encoder_functions import PChemPropEncoder
from .shared_eval_funcs import optuna_classification_target_prc, write_res_to_file


def il6_eval(project_dir):
    """Runs the evals for the Cognano dataset for xgboost."""
    filtered_seqs = preprocess_il6(project_dir)

    xgboost_eval(project_dir, filtered_seqs, PChemPropEncoder())



def xgboost_eval(project_dir, filtered_seqs, encoder):
    """Runs train-test split evaluations."""

    fit_times, prc_scores = [], []

    for i in range(5):
        trainx, testx, trainy, testy = build_traintest_set(filtered_seqs, i,
                num_desired = 1636, encoder = encoder)

        timestamp = time.time()

        sampler = optuna.samplers.TPESampler(seed=123)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(lambda trial: optuna_classification_target_prc(trial, trainx,
            trainy), n_trials=100)
        trial = study.best_trial
        params = trial.params
        xgboost_model = XGBClassifier(**params)
        xgboost_model.fit(trainx, trainy)

        fit_times.append(time.time() - timestamp)

        probs = xgboost_model.predict_proba(testx)

        prc_scores.append(average_precision_score(testy, probs[:,1]))

        print(f"****\n{prc_scores[-1]}\n******")

    write_res_to_file(project_dir, "IL6", "XGBoost", type(encoder).__name__,
            fit_times = fit_times, auc_prc_scores = prc_scores)


def build_traintest_set(filtered_data, random_seed, num_desired,
        encoder):
    """Builds a training and test set by undersampling the negatives.
    The data prep procedure adopted by Barton et al. for this dataset
    is a little unique in some ways (very unusual criteria for datapoint
    exclusion for example) -- we have used it however to
    try to ensure a close comparison."""
    labels, stacked_seqs = [], []

    for prepped_seq, label_arr in filtered_data.items():
        stacked_seqs.append(prepped_seq)
        labels.append(np.sum(label_arr))

    positive_idx = np.where(np.array(labels)==1)[0].tolist()
    num_neg = num_desired - len(positive_idx)
    if num_neg <= 0:
        raise RuntimeError("Requested a number of negatives <= 0!")

    neg_idx = np.where(np.array(labels)==0)[0].tolist()

    random.seed(random_seed)
    random.shuffle(neg_idx)

    idx = positive_idx + neg_idx[:num_neg]
    random.shuffle(idx)

    retained_labels = [labels[idn] for idn in idx]
    retained_seqs = [stacked_seqs[idn] for idn in idx]

    if len(retained_labels) != len(retained_seqs) or len(retained_labels) != \
            num_desired:
        raise RuntimeError("Error in seq processing.")
    if np.sum(retained_labels) != len(positive_idx):
        raise RuntimeError("Error in seq processing.")

    all_y = np.array(retained_labels)
    all_x = encoder.encode_variable_length(retained_seqs)
    if len(all_x.shape) > 2:
        all_x = all_x.reshape((all_x.shape[0], all_x.shape[1] *
            all_x.shape[2]))

    cutoff_val, cutoff_test = int(0.8 * num_desired), \
            int(0.9 * num_desired)

    trainy, testy = all_y[:cutoff_val], all_y[cutoff_val:cutoff_test]
    trainx, testx = all_x[:cutoff_val,:], all_x[cutoff_val:cutoff_test,:]
    return trainx, testx, trainy, testy
