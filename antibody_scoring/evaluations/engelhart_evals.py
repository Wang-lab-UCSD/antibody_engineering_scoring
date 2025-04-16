"""Contains code for evaluating xgboost and xgpr on the Engelhart dataset."""
import time
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from xgboost import XGBRegressor
from xGPR import xGPRegression, build_regression_dataset
from xGPR import xGPDiscriminant, build_classification_dataset, tune_classifier_optuna
import numpy as np
from ..data_prep.data_processing import preprocess_engelhart
from ..data_prep.seq_encoder_functions import OneHotEncoder, PFAStandardEncoder, PChemPropEncoder
from .shared_eval_funcs import optuna_regression, write_res_to_file


def engelhart_eval(project_dir):
    """Runs the evals for the Engelhart dataset for both xgpr and xgboost."""
    fixed_len_seqs, yvalues, _, _, seq_unaligned, seq_lengths = \
            preprocess_engelhart(project_dir, False)
    xgpr_discriminant_eval(project_dir, seq_unaligned, seq_lengths,
            yvalues, OneHotEncoder(), kernel="Conv1dRBF")

    xgboost_eval(project_dir, fixed_len_seqs, yvalues,
                PChemPropEncoder())

    xgpr_eval(project_dir, seq_unaligned, np.array(seq_lengths),
            yvalues, OneHotEncoder())
    xgpr_eval(project_dir, seq_unaligned, np.array(seq_lengths),
            yvalues, PFAStandardEncoder())



def xgpr_eval(project_dir, seq_unaligned, slengths, yvalues, encoder,
                        kernel="Conv1dRBF"):
    """Runs train-test split evaluations."""
    xvalues = encoder.encode_variable_length(seq_unaligned).astype(np.float32)
    auc_roc_scores, auc_prc_scores, r2_scores, fit_times = [], [], [], []

    for i in range(5):
        timestamp = time.time()
        trainx, trainy, trainlen, testx, testy, testlen = get_tt_split(xvalues,
                yvalues, i, slengths)

        if "Conv" in kernel:
            regdata = build_regression_dataset(trainx, trainy, trainlen, chunk_size=2000)
        else:
            regdata = build_regression_dataset(trainx, trainy, chunk_size=2000)

        xgp = xGPRegression(num_rffs = 2048, variance_rffs = 12, kernel_choice = kernel,
                kernel_settings = {"intercept":True, "conv_width":9,
                    "averaging":"sqrt"}, device="gpu")

        _ = xgp.tune_hyperparams_crude(regdata)
        xgp.num_rffs = 4096

        xgp.tune_hyperparams(regdata)
        xgp.num_rffs = 16384
        xgp.fit(regdata, suppress_var = True)
        fit_times.append(time.time() - timestamp)

        if "Conv" in kernel:
            preds = xgp.predict(testx, testlen)
        else:
            preds = xgp.predict(testx)

        r2_scores.append(r2_score(testy, preds))
        valcat = testy.copy()
        valcat[valcat<=3]=0
        valcat[valcat>3]=1
        auc_roc_scores.append(roc_auc_score(valcat, preds))
        auc_prc_scores.append(average_precision_score(valcat, preds))

    write_res_to_file(project_dir, "barton", f"xGPR_{kernel}", type(encoder).__name__,
            r2_scores = r2_scores, auc_roc_scores = auc_roc_scores,
            auc_prc_scores = auc_prc_scores, fit_times = fit_times)




def xgpr_discriminant_eval(project_dir, seq_unaligned, slengths,
        yvalues, encoder, kernel="Conv1dRBF"):
    """Runs train-test split evaluations."""
    xvalues = encoder.encode_variable_length(seq_unaligned).astype(np.float32)
    auc_roc_scores, auc_prc_scores, fit_times = [], [], []

    # Convert the input real values to categorical values.
    ycat = yvalues.copy()
    ycat[ycat<=3] = 0
    ycat[ycat>3] = 1
    ycat = ycat.astype(np.int64)

    for i in range(5):
        timestamp = time.time()
        trainx, trainy, trainlen, testx, testy, testlen = get_tt_split(xvalues,
                ycat, i, slengths)
        trainx, val_x, trainy, val_y, trainlen, val_len = train_test_split(trainx,
                trainy, trainlen, test_size=0.2, shuffle=False)

        classdata = build_classification_dataset(trainx, trainy,
                    trainlen, chunk_size=2000)
        val_data = build_classification_dataset(val_x, val_y, val_len,
                chunk_size=2000)

        xgp = xGPDiscriminant(num_rffs = 4096,
                kernel_choice = kernel, kernel_settings =
                    {"intercept":True, "conv_width":9,
                    "averaging":"sqrt"}, device="cuda")

        xgp, _, _ = tune_classifier_optuna(classdata, val_data,
                xgp, fit_mode="exact", eval_metric="cross_entropy",
                max_iter=50)
        xgp.num_rffs = 8192
        xgp.fit(classdata)
        fit_times.append(time.time() - timestamp)

        preds = xgp.predict(testx, testlen)

        auc_roc_scores.append(roc_auc_score(testy, preds[:,-1]))
        auc_prc_scores.append(average_precision_score(testy, preds[:,-1]))

    write_res_to_file(project_dir, "barton", f"xGPDiscriminant_{kernel}",
            type(encoder).__name__, auc_roc_scores = auc_roc_scores,
            auc_prc_scores = auc_prc_scores,
            fit_times = fit_times)




def xgboost_eval(project_dir, input_seqs, yvalues,
        encoder):
    """Runs train-test split evaluations."""
    xvalues = encoder.encode_variable_length(input_seqs)
    if len(xvalues.shape) > 2:
        xvalues = xvalues.reshape((xvalues.shape[0], xvalues.shape[1] * xvalues.shape[2]))

    auc_roc_scores, auc_prc_scores, r2_scores, fit_times = [], [], [], []

    for i in range(5):
        timestamp = time.time()
        trainx, trainy, _, testx, testy, _ = get_tt_split(xvalues, yvalues, i)

        sampler = optuna.samplers.TPESampler(seed=123)
        study = optuna.create_study(sampler=sampler, direction='minimize')
        study.optimize(lambda trial: optuna_regression(trial, trainx, trainy), n_trials=100)
        xgboost_model = XGBRegressor(**study.best_trial.params)
        xgboost_model.fit(trainx, trainy)

        fit_times.append(time.time() - timestamp)

        preds = xgboost_model.predict(testx)

        r2_scores.append(r2_score(testy, preds))
        valcat = testy.copy()
        valcat[valcat<=3]=0
        valcat[valcat>3]=1
        auc_roc_scores.append(roc_auc_score(valcat, preds))
        auc_prc_scores.append(average_precision_score(valcat, preds))

    write_res_to_file(project_dir, "barton", "XGBoost", type(encoder).__name__,
            r2_scores = r2_scores, auc_roc_scores = auc_roc_scores,
            auc_prc_scores = auc_prc_scores, fit_times = fit_times)





def get_tt_split(xdata, ydata, random_seed, slengths = None):
    """Splits the input data up into train and test."""
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(ydata.shape[0])
    trainidx, testidx, _ = idx[:41787], idx[41787:41787+5159], idx[41787+5159:]

    trainx, trainy = xdata[trainidx,...], ydata[trainidx]
    testx, testy = xdata[testidx,...], ydata[testidx]
    if slengths is None:
        return trainx, trainy, None, testx, testy, None

    slengths = np.array(slengths)
    trainlen, testlen = slengths[trainidx], slengths[testidx]
    return trainx, trainy, trainlen, testx, testy, testlen
