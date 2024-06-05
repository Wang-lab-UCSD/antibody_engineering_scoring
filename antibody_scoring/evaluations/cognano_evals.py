"""Contains code for evaluating xgboost on the Cognano dataset."""
import time
import optuna
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import numpy as np
from ..data_prep.data_processing import preprocess_cognano
from ..data_prep.seq_encoder_functions import PChemPropEncoder
from .shared_eval_funcs import optuna_classification_target_f1, write_res_to_file


def cognano_eval(project_dir):
    """Runs the evals for the Cognano dataset for xgboost."""
    train_data, test_data = preprocess_cognano(project_dir)

    xgboost_eval(project_dir, train_data, test_data, PChemPropEncoder())



def xgboost_eval(project_dir, train_data, test_data, encoder):
    """Runs train-test split evaluations."""
    trainx = encoder.encode_variable_length(train_data[0])
    if len(trainx.shape) > 2:
        trainx = trainx.reshape((trainx.shape[0], trainx.shape[1] *
            trainx.shape[2]))
    trainx = np.hstack([trainx, train_data[1].astype(np.float32)])

    testx = encoder.encode_variable_length(test_data[0])
    if len(testx.shape) > 2:
        testx = testx.reshape((testx.shape[0], testx.shape[1] *
            testx.shape[2]))
    testx = np.hstack([testx, test_data[1].astype(np.float32)])

    trainy, testy = np.array(train_data[2]), np.array(test_data[2])
    timestamp = time.time()

    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(lambda trial: optuna_classification_target_f1(trial, trainx,
        trainy), n_trials=100)
    trial = study.best_trial
    params = trial.params
    xgboost_model = XGBClassifier(**params)
    xgboost_model.fit(trainx, trainy)

    fit_times = [time.time() - timestamp]

    preds = xgboost_model.predict(testx)
    probs = xgboost_model.predict_proba(testx)

    precision_scores = [precision_score(testy, preds)]
    recall_scores = [recall_score(testy, preds)]
    f1_scores = [f1_score(testy, preds)]
    auc_prc = [average_precision_score(testy, probs[:,1])]
    accuracy_scores = [accuracy_score(testy, preds)]

    print(f"****\n{accuracy_scores[0]}\n******")

    write_res_to_file(project_dir, "Cognano", "XGBoost", type(encoder).__name__,
            fit_times = fit_times, precision_scores = precision_scores,
            recall_scores = recall_scores, f1_scores = f1_scores, auc_prc_scores = auc_prc,
            accuracy_scores = accuracy_scores)
