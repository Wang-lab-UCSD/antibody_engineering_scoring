"""Contains code for evaluating xgboost on the Desautels dataset."""
import time
import optuna
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import numpy as np
from ..data_prep.data_processing import preprocess_desautels
from ..data_prep.seq_encoder_functions import PChemPropEncoder, OneHotEncoder, PFAStandardEncoder
from .shared_eval_funcs import optuna_regression, write_res_to_file


def desautels_eval(project_dir):
    """Runs the evals for the Desautels dataset for xgboost."""
    pchem_prop, ohenc, pfaenc = PChemPropEncoder(), OneHotEncoder(), PFAStandardEncoder()
    seqs, yvalues = preprocess_desautels(project_dir)

    xgboost_eval(project_dir, seqs, yvalues, pchem_prop)

    



def xgboost_eval(project_dir, fixed_len_seqs, all_yvalues,
        encoder):
    """Runs train-test split evaluations."""
    all_xvalues = encoder.encode_variable_length(fixed_len_seqs)
    all_xvalues = all_xvalues.reshape((all_xvalues.shape[0], all_xvalues.shape[1] * all_xvalues.shape[2]))

    spearman_scores, fit_times = [], []

    for i in range(5):
        subset_spearman_scores = []

        for ycol in range(5):
            yvalues = all_yvalues[:,ycol]
            # One column contains some nan values. If we are working with that column,
            # filter out any problem datapoints.
            xvalues = all_xvalues

            if np.isnan(yvalues).sum() > 0:
                idx = ~np.isnan(yvalues)
                xvalues, yvalues = xvalues[idx,:].copy(), yvalues[idx].copy()

            timestamp = time.time()
            trainx, trainy, testx, testy = get_tt_split(xvalues, yvalues, i,
                train_percent = 0.2)

            sampler = optuna.samplers.TPESampler(seed=123)
            study = optuna.create_study(sampler=sampler, direction='minimize')
            study.optimize(lambda trial: optuna_regression(trial, trainx, trainy), n_trials=100)
            trial = study.best_trial
            params = trial.params
            xgboost_model = XGBRegressor(**params)
            xgboost_model.fit(trainx, trainy)

            fit_times.append(time.time() - timestamp)

            preds = xgboost_model.predict(testx)

            subset_spearman_scores.append(spearmanr(testy, preds)[0])
            print(f"****\n{subset_spearman_scores[-1]}\n******")

        spearman_scores.append(np.mean(subset_spearman_scores))

    write_res_to_file(project_dir, "Desautels", "XGBoost", type(encoder).__name__,
            fit_times = fit_times, spearman_scores = spearman_scores,
            train_percent = "0.2")


def xgpr_rbf_eval(project_dir, seq_unaligned, slengths, yvalues, encoder,
        regression_only = False):
    """Runs train-test evaluations using a train fraction of 0.2 (to compare with
    Singh et al.)"""
    xvalues = encoder.encode_variable_length(seq_unaligned).astype(np.float32)

    auc_roc_scores, auc_prc_scores, r2_scores, fit_times, mae_scores = [], [], [], [], []
    if regression_only:
        study_name = "deutschmann"
    else:
        study_name = "barton"

    for i in range(5):
        timestamp = time.time()
        trainx, trainy, trainlen, testx, testy, testlen = get_tt_split(xvalues, yvalues, i,
                study_name, slengths)

        regdata = build_regression_dataset(trainx, trainy, trainlen, chunk_size=2000)
        xgp = xGPRegression(num_rffs = 2048, variance_rffs = 12, kernel_choice = "Conv1dRBF",
                kernel_settings = {"intercept":True, "conv_width":9,
                    "averaging":"sqrt"}, device="gpu")

        _ = xgp.tune_hyperparams_crude(regdata)
        xgp.num_rffs = 4096

        xgp.tune_hyperparams(regdata)
        xgp.num_rffs = 16384
        xgp.fit(regdata, suppress_var = True)
        fit_times.append(time.time() - timestamp)
        time.sleep(4)

        preds = xgp.predict(testx, testlen)
        print(r2_score(testy, preds))
        if regression_only:
            r2_scores.append(r2_score(testy, preds))
            mae_scores.append(np.mean(np.abs(testy - preds)))
        else:
            valcat = testy.copy()
            valcat[valcat<=3]=0
            valcat[valcat>3]=1
            auc_roc_scores.append(roc_auc_score(valcat, preds))
            auc_prc_scores.append(average_precision_score(valcat, preds))

    if not regression_only:
        write_res_to_file(project_dir, "barton", "xGPR_Conv1dRBF", type(encoder).__name__,
            r2_scores, mae_scores, auc_roc_scores,
            auc_prc_scores, fit_times)
    else:
        write_res_to_file(project_dir, "deutschmann", "xGPR_Conv1dRBF", type(encoder).__name__,
            r2_scores, mae_scores, auc_roc_scores,
            auc_prc_scores, fit_times)






def get_tt_split(xdata, ydata, random_seed, train_percent = 0.2):
    """Splits the input data up into train and test."""
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(ydata.shape[0])
    cutoff = int(train_percent * idx.shape[0])
    trainidx, testidx = idx[:cutoff], idx[cutoff:]

    trainx, trainy = xdata[trainidx,...], ydata[trainidx]
    testx, testy = xdata[testidx,...], ydata[testidx]
    return trainx, trainy, testx, testy
