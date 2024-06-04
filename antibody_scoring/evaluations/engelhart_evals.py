"""Contains code for evaluating xgboost and xgpr on the Engelhart dataset."""
import time
import optuna
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from xgboost import XGBRegressor
from xGPR import xGPRegression, build_regression_dataset, FastConv1d
import numpy as np
from ..data_prep.data_processing import preprocess_engelhart
from ..data_prep.seq_encoder_functions import OneHotEncoder, PFAStandardEncoder, PChemPropEncoder
from .shared_eval_funcs import optuna_regression, write_res_to_file


def engelhart_eval(project_dir):
    """Runs the evals for the Engelhart dataset for both xgpr and xgboost."""
    #eval_group(project_dir, False)
    eval_group(project_dir, True)


def eval_group(project_dir, regression_only = False):
    """Performs the evaluations for a given setting of regression only as false or true."""
    pchem, ohenc, pfaenc = PChemPropEncoder(), OneHotEncoder(), PFAStandardEncoder()
    fixed_len_seqs, yvalues, _, _, seq_unaligned, seq_lengths = \
            preprocess_engelhart(project_dir, regression_only)

    xgboost_eval(project_dir, fixed_len_seqs, yvalues, pchem,
            regression_only = True)

    xgpr_conv1drbf_eval(project_dir, seq_unaligned, np.array(seq_lengths),
            yvalues, ohenc, regression_only)
    xgpr_conv1drbf_eval(project_dir, seq_unaligned, np.array(seq_lengths),
            yvalues, pfaenc, regression_only)



def xgpr_conv1drbf_eval(project_dir, seq_unaligned, slengths, yvalues, encoder,
        regression_only = False):
    """Runs train-test split evaluations using either regression only settings
    (for comparison with Deutschmann et al) or classification (for comparison
    with Barton et al)."""
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




def xgboost_eval(project_dir, fixed_len_seqs, yvalues,
        encoder, regression_only = False):
    """Runs train-test split evaluations using either regression only settings
    (for comparison with Deutschmann et al) or classification (for comparison
    with Barton et al)."""
    xvalues = encoder.encode_variable_length(fixed_len_seqs)
    xvalues = xvalues.reshape((xvalues.shape[0], xvalues.shape[1] * xvalues.shape[2]))

    auc_roc_scores, auc_prc_scores, r2_scores, fit_times, mae_scores = [], [], [], [], []
    if regression_only:
        study_name = "deutschmann"
    else:
        study_name = "barton"

    for i in range(5):
        timestamp = time.time()
        trainx, trainy, testx, testy = get_tt_split(xvalues, yvalues, i, study_name)

        sampler = optuna.samplers.TPESampler(seed=123)
        study = optuna.create_study(sampler=sampler, direction='minimize')
        study.optimize(lambda trial: optuna_regression(trial, trainx, trainy), n_trials=100)
        trial = study.best_trial
        params = trial.params
        xgboost_model = XGBRegressor(**params)
        xgboost_model.fit(trainx, trainy)

        fit_times.append(time.time() - timestamp)

        preds = xgboost_model.predict(testx)

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
        write_res_to_file(project_dir, "barton", "XGBoost", type(encoder).__name__,
            r2_scores, mae_scores, auc_roc_scores,
            auc_prc_scores, fit_times)
    else:
        write_res_to_file(project_dir, "deutschmann", "XGBoost", type(encoder).__name__,
            r2_scores, mae_scores, auc_roc_scores,
            auc_prc_scores, fit_times)


def get_tt_split(xdata, ydata, random_seed, study = "barton", slengths = None):
    """Splits the input data up into train and test."""
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(ydata.shape[0])
    print(study)
    if study == "barton":
        trainidx, testidx, _ = idx[:41787], idx[41787:41787+5159], idx[41787+5159:]
    elif study == "deutschmann":
        cutoff = int(0.8 * idx.shape[0])
        trainidx, testidx = idx[:cutoff], idx[cutoff:]
    else:
        raise RuntimeError("Unreognized study supplied.")

    trainx, trainy = xdata[trainidx,...], ydata[trainidx]
    testx, testy = xdata[testidx,...], ydata[testidx]
    if slengths is None:
        return trainx, trainy, testx, testy

    trainlen, testlen = slengths[trainidx], slengths[testidx]
    return trainx, trainy, trainlen, testx, testy, testlen
