"""Contains code for evaluating xgboost on the Desautels dataset."""
import time
import optuna
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from xGPR import xGPRegression, build_regression_dataset
import numpy as np
from ..data_prep.data_processing import preprocess_desautels
from ..data_prep.seq_encoder_functions import PChemPropEncoder, OneHotEncoder, PFAStandardEncoder
from .shared_eval_funcs import optuna_regression, write_res_to_file


def desautels_eval(project_dir):
    """Runs the evals for the Desautels dataset for xgboost."""
    pchem_prop, ohenc, pfaenc = PChemPropEncoder(), OneHotEncoder(), PFAStandardEncoder()
    seqs, yvalues = preprocess_desautels(project_dir)

    #xgboost_eval(project_dir, seqs, yvalues, pchem_prop)
    #xgpr_rbf_eval(project_dir, seqs, yvalues, ohenc)
    #xgpr_rbf_eval(project_dir, seqs, yvalues, pfaenc)

    #xgboost_eval(project_dir, seqs, yvalues, pchem_prop, 0.05)
    #xgpr_rbf_eval(project_dir, seqs, yvalues, ohenc, 0.05)
    #xgpr_rbf_eval(project_dir, seqs, yvalues, pfaenc, 0.05)

    xgboost_eval(project_dir, seqs, yvalues, pchem_prop, 0.01)
    xgpr_rbf_eval(project_dir, seqs, yvalues, ohenc, 0.01)
    xgpr_rbf_eval(project_dir, seqs, yvalues, pfaenc, 0.01)



def xgboost_eval(project_dir, fixed_len_seqs, all_yvalues,
        encoder, train_fraction = 0.2):
    """Runs train-test split evaluations."""
    all_xvalues = encoder.encode_variable_length(fixed_len_seqs)
    all_xvalues = all_xvalues.reshape((all_xvalues.shape[0],
        all_xvalues.shape[1] * all_xvalues.shape[2]))

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
                train_percent = train_fraction)

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
            train_percent = f"{train_fraction}")


def xgpr_rbf_eval(project_dir, seqs, all_yvalues, encoder, train_fraction = 0.2):
    """Runs train-test evaluations using a specified train fraction (to compare with
    Singh et al.)"""
    all_xvalues = encoder.encode_variable_length(seqs).astype(np.float32)
    all_xvalues = all_xvalues.reshape((all_xvalues.shape[0],
        all_xvalues.shape[1] * all_xvalues.shape[2]))

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
                train_percent = train_fraction)

            regdata = build_regression_dataset(trainx, trainy, chunk_size=2000)
            xgp = xGPRegression(num_rffs = 2048, variance_rffs = 12, kernel_choice = "RBF",
                kernel_settings = {"intercept":True}, device="gpu")

            _ = xgp.tune_hyperparams_crude(regdata)
            xgp.num_rffs = 4096

            xgp.tune_hyperparams(regdata)
            xgp.num_rffs = 16384
            xgp.fit(regdata, suppress_var = True)
            fit_times.append(time.time() - timestamp)
            time.sleep(4)

            preds = xgp.predict(testx)
            subset_spearman_scores.append(spearmanr(testy, preds)[0])
            print(f"****\n{subset_spearman_scores[-1]}\n******")

        spearman_scores.append(np.mean(subset_spearman_scores))


    write_res_to_file(project_dir, "Desautels", "xGPR_RBF", type(encoder).__name__,
            fit_times = fit_times, spearman_scores = spearman_scores,
            train_percent = f"{train_fraction}")






def get_tt_split(xdata, ydata, random_seed, train_percent = 0.2):
    """Splits the input data up into train and test."""
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(ydata.shape[0])
    cutoff = int(train_percent * idx.shape[0])
    trainidx, testidx = idx[:cutoff], idx[cutoff:]

    trainx, trainy = xdata[trainidx,...], ydata[trainidx]
    testx, testy = xdata[testidx,...], ydata[testidx]
    return trainx, trainy, testx, testy
