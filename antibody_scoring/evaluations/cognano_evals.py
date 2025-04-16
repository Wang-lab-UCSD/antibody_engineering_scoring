"""Contains code for evaluating xgboost on the Cognano dataset."""
import time
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from xGPR import xGPDiscriminant, build_classification_dataset, cv_tune_classifier_optuna
from xgboost import XGBClassifier
import numpy as np
from ..data_prep.data_processing import preprocess_cognano
from ..data_prep.seq_encoder_functions import PChemPropEncoder, PFAStandardEncoder
from .shared_eval_funcs import optuna_classification, write_res_to_file


def cognano_eval(project_dir):
    """Runs the evals for the Cognano dataset for xgboost."""
    train_data, test_data = preprocess_cognano(project_dir)

    #xgboost_eval(project_dir, train_data, test_data, PChemPropEncoder())
    xgpr_discriminant_eval(project_dir, train_data, test_data,
        PFAStandardEncoder("raw"))



def xgboost_eval(project_dir, train_data, test_data, encoder):
    """Runs train-test split evaluations. In this case the
    test set is fixed, but we can still vary the random seed
    used for tuning and fitting."""
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

    fit_times, auc_prc = [], []
    precision_scores, accuracy_scores, f1_scores, recall_scores = [], [], [], []

    for i in range(5):
        timestamp = time.time()

        sampler = optuna.samplers.TPESampler(seed=i)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(lambda trial: optuna_classification(trial, trainx,
            trainy, target="f1"), n_trials=100)
        trial = study.best_trial
        params = trial.params
        params["random_state"] = i
        xgboost_model = XGBClassifier(**params)
        xgboost_model.fit(trainx, trainy)

        fit_times.append(time.time() - timestamp)

        preds = xgboost_model.predict(testx)
        probs = xgboost_model.predict_proba(testx)

        precision_scores.append(precision_score(testy, preds))
        recall_scores.append(recall_score(testy, preds))
        f1_scores.append(f1_score(testy, preds))
        auc_prc.append(average_precision_score(testy, probs[:,1]))
        accuracy_scores.append(accuracy_score(testy, preds))

    print(f"****\n{accuracy_scores[0]}\n******")

    write_res_to_file(project_dir, "Cognano", "XGBoost", type(encoder).__name__,
            fit_times = fit_times, precision_scores = precision_scores,
            recall_scores = recall_scores, f1_scores = f1_scores, auc_prc_scores = auc_prc,
            accuracy_scores = accuracy_scores)



def xgpr_discriminant_eval(project_dir, train_data, test_data,
        encoder):
    """Runs train-test split evaluations."""
    fit_times, auc_prc = [], []
    precision_scores, accuracy_scores, f1_scores, recall_scores = [], [], [], []

    trainx = encoder.encode_variable_length(train_data[0])
    trainx = trainx.reshape((trainx.shape[0], trainx.shape[1] * trainx.shape[2]))
    trainx = np.hstack([trainx, train_data[1].astype(np.float32)])

    testx = encoder.encode_variable_length(test_data[0])
    testx = testx.reshape((testx.shape[0], testx.shape[1] * testx.shape[2]))
    testx = np.hstack([testx, test_data[1].astype(np.float32)])

    trainy, testy = np.array(train_data[2]), np.array(test_data[2])

    # For now we are only running this once. TODO: Add repeats with
    # different random seeds (test set is always the same here).
    for i in range(1):
        timestamp = time.time()

        xgp = xGPDiscriminant(num_rffs = 4096,
                kernel_choice = "MiniARD", kernel_settings =
                    {"intercept":True, "conv_width":9,
                    "split_points":[4011]}, device="cuda")

        xgp, _, _ = cv_tune_classifier_optuna(trainx, trainy,
                xgp, fit_mode="exact", eval_metric="aucprc",
                max_iter=50)
        xgp.num_rffs = 16384
        classdata = build_classification_dataset(trainx, trainy)

        xgp.fit(classdata)
        fit_times.append(time.time() - timestamp)

        preds = xgp.predict(testx)
        class_preds = np.argmax(preds, axis=1)

        precision_scores.append(precision_score(testy, class_preds))
        recall_scores.append(recall_score(testy, class_preds))
        f1_scores.append(f1_score(testy, class_preds))
        auc_prc.append(average_precision_score(testy, preds[:,-1]))
        accuracy_scores.append(accuracy_score(testy, class_preds))
        print(f"********Latest score: {auc_prc[-1]}*********")

    write_res_to_file(project_dir, "Cognano", "xGPDiscriminant_RBF", type(encoder).__name__,
            fit_times = fit_times, precision_scores = precision_scores,
            recall_scores = recall_scores, f1_scores = f1_scores, auc_prc_scores = auc_prc,
            accuracy_scores = accuracy_scores)
