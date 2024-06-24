"""Contains code for evaluating xgboost on the Mason dataset."""
import time
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import numpy as np
from ..data_prep.data_processing import preprocess_mason
from ..data_prep.seq_encoder_functions import PChemPropEncoder, IntegerEncoder, AbLangEncoder
from .shared_eval_funcs import optuna_classification, write_res_to_file, build_bayes_classifier


def mason_eval(project_dir):
    """Runs the evals for the Mason dataset for xgboost."""
    cdr_seqs, yvalues = preprocess_mason(project_dir)

    #xgboost_eval(project_dir, cdr_seqs, yvalues, PChemPropEncoder)
    #catmix_eval(project_dir, cdr_seqs, yvalues, IntegerEncoder())
    xgboost_eval(project_dir, cdr_seqs, yvalues, AbLangEncoder())



def xgboost_eval(project_dir, fixed_len_seqs, yvalues,
        encoder):
    """Runs train-test split evaluations."""
    if isinstance(encoder, AbLangEncoder):
        xvalues = encoder.encode_mason_data(fixed_len_seqs)
    else:
        xvalues = encoder.encode_variable_length(fixed_len_seqs)
        xvalues = xvalues.reshape((xvalues.shape[0], xvalues.shape[1] * xvalues.shape[2]))

    auc_roc_scores, auc_prc_scores, fit_times = [], [], []

    for i in range(5):
        timestamp = time.time()
        trainx, trainy, testx, testy = get_tt_split(xvalues, yvalues, i)

        sampler = optuna.samplers.TPESampler(seed=123)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(lambda trial: optuna_classification(trial,
                                        trainx, trainy), n_trials=100)
        trial = study.best_trial
        params = trial.params
        xgboost_model = XGBClassifier(**params)
        xgboost_model.fit(trainx, trainy)

        fit_times.append(time.time() - timestamp)

        preds = xgboost_model.predict_proba(testx)[:,1]

        auc_roc_scores.append(roc_auc_score(testy, preds))
        auc_prc_scores.append(average_precision_score(testy, preds))

    write_res_to_file(project_dir, "mason", "XGBoost", type(encoder).__name__,
            auc_roc_scores = auc_roc_scores, auc_prc_scores = auc_prc_scores,
            fit_times = fit_times)



def catmix_eval(project_dir, fixed_len_seqs, yvalues, encoder):
    """Trains a mixture of categorical distributions on both
    positives and negatives; uses a simple Bayes' classifier.
    Should only be used with the integer encoder."""
    xvalues = encoder.encode(fixed_len_seqs)

    auc_roc_scores, auc_prc_scores, fit_times = [], [], []

    for i in range(5):
        timestamp = time.time()
        trainx, trainy, testx, testy = get_tt_split(xvalues, yvalues, i)
        preds = build_bayes_classifier(trainx, trainy,
                testx, use_aic = False, num_possible_items = 21)

        fit_times.append(time.time() - timestamp)

        auc_roc_scores.append(roc_auc_score(testy, preds))
        auc_prc_scores.append(average_precision_score(testy, preds))

    write_res_to_file(project_dir, "mason", "BayesCatmix", type(encoder).__name__,
            auc_roc_scores = auc_roc_scores, auc_prc_scores = auc_prc_scores,
            fit_times = fit_times)




def get_tt_split(xdata, ydata, random_seed):
    """Splits the input data up into train and test."""
    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(ydata.shape[0])
    cutoff = int(0.8 * idx.shape[0])
    cutoff_test = int(0.9 * idx.shape[0])
    # Notice that we don't bother with the validation set here...
    # we don't need it, therefore we leave it unused.
    trainidx, testidx = idx[:cutoff], idx[cutoff_test:]

    trainx, trainy = xdata[trainidx,...], ydata[trainidx]
    testx, testy = xdata[testidx,...], ydata[testidx]
    return trainx, trainy, testx, testy
