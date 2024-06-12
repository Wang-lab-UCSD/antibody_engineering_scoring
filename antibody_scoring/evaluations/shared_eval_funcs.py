"""Contains shared functions for hyperparameter tuning and result handling shared by
all of the eval modules."""
import os
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier


def write_res_to_file(project_dir, dataset_name, model_type, encoding,
        r2_scores = [], auc_roc_scores = [], auc_prc_scores = [], fit_times = [],
        spearman_scores = [], train_percent = "NA", precision_scores = [],
        recall_scores = [], f1_scores = [], accuracy_scores = []):
    """Writes the result of an evaluation to file, accommodating all
    the different metrics we will need to use, some of which are only needed
    for certain datasets."""
    output_fname = os.path.join(project_dir, "results", "evaluation_results.txt")
    if not os.path.exists(output_fname):
        with open(output_fname, "w+", encoding="utf-8") as fhandle:
            fhandle.write("Dataset,Model,Encoding,R^2,AUC_ROC,AUC_PRC,Spearmanr,"
                    "Fit_time,Train_percent,Precision_scores,Recall_scores,F1_scores,"
                    "Accuracy_scores\n")

    with open(output_fname, "a", encoding="utf-8") as fhandle:
        fhandle.write(f"{dataset_name},{model_type},{encoding},"
                f"{convert_scores_to_stats(r2_scores)},"
                f"{convert_scores_to_stats(auc_roc_scores)},"
                f"{convert_scores_to_stats(auc_prc_scores)},"
                f"{convert_scores_to_stats(spearman_scores)},"
                f"{convert_scores_to_stats(fit_times)},"
                f"{train_percent},"
                f"{convert_scores_to_stats(precision_scores)},"
                f"{convert_scores_to_stats(recall_scores)},"
                f"{convert_scores_to_stats(f1_scores)},"
                f"{convert_scores_to_stats(accuracy_scores)}\n")




def convert_scores_to_stats(scores):
    """Takes a list of scores and if empty returns NA, otherwise
    converts it to a mean and 95% CI."""
    if len(scores) == 0:
        return "NA"
    if len(scores) == 1:
        return f"{scores[0]}"
    conf_int_h = norm.interval(0.95, loc=0,
            scale=np.std(scores, ddof=1) / np.sqrt(len(scores)))[1]
    return f"{np.mean(scores)}+/-{conf_int_h}"




def optuna_regression(trial, trainx, trainy, target="mae",
        min_colsample_bound = 0.05):
    """Regression objective for Optuna tuning."""
    kf = KFold(n_splits=5)

    kf.get_n_splits(trainx)

    params = {
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 0.1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', min_colsample_bound,
                1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'eval_metric': 'mlogloss',
            'device': "cuda"
    }

    optuna_model = XGBRegressor(**params)
    results = []
    for (train_index, test_index) in kf.split(trainx):
        subtrainx, subvalidx = trainx[train_index,...], trainx[test_index,...]
        subtrainy, subvalidy = trainy[train_index], trainy[test_index]

        optuna_model.fit(subtrainx, subtrainy)
        y_pred = optuna_model.predict(subvalidx)
        if target == "mae":
            score = np.mean(np.abs(subvalidy - y_pred))
        elif target == "spearman":
            score = spearmanr(subvalidy, y_pred)[0]
        else:
            raise RuntimeError("Unsupported score requested from hyperparameter "
                    "tuning function.")
        results.append(score)

    return np.mean(results)



def optuna_classification(trial, trainx, trainy):
    """Optuna objective for classification when the dataset is
    reasonably balanced and/or AUC-ROC is a good metric."""
    kf = KFold(n_splits=5)

    kf.get_n_splits(trainx)

    params = {
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 0.1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.05, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'eval_metric': 'mlogloss',
            'device': "cuda"
    }

    optuna_model = XGBClassifier(**params)
    results = []
    for (train_index, test_index) in kf.split(trainx):
        subtrainx, subvalidx = trainx[train_index,...], trainx[test_index,...]
        subtrainy, subvalidy = trainy[train_index], trainy[test_index]

        optuna_model.fit(subtrainx, subtrainy)
        y_pred = optuna_model.predict_proba(subvalidx)[:,1]
        score = roc_auc_score(subvalidy, y_pred)
        results.append(score)

    return np.mean(results)


def optuna_classification_target_f1(trial, trainx, trainy):
    """Optuna target for when the dataset is imbalanced or AUC-ROC would
    not be a good metric; optimizes for improved F1 instead."""
    kf = KFold(n_splits=5)

    kf.get_n_splits(trainx)

    params = {
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 0.1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.05, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'eval_metric': 'mlogloss',
            'device': "cuda"
    }

    optuna_model = XGBClassifier(**params)
    results = []
    for (train_index, test_index) in kf.split(trainx):
        subtrainx, subvalidx = trainx[train_index,...], trainx[test_index,...]
        subtrainy, subvalidy = trainy[train_index], trainy[test_index]

        optuna_model.fit(subtrainx, subtrainy)
        y_pred = optuna_model.predict(subvalidx)
        score = f1_score(subvalidy, y_pred)
        results.append(score)

    return np.mean(results)


def optuna_classification_target_prc(trial, trainx, trainy):
    """Optuna target for when the dataset is imbalanced or AUC-ROC would
    not be a good metric; optimizes for improved AUC-PRC instead."""
    kf = KFold(n_splits=5)

    kf.get_n_splits(trainx)

    params = {
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 0.1, 10),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.05, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'eval_metric': 'mlogloss',
            'device': "cuda"
    }

    optuna_model = XGBClassifier(**params)
    results = []
    for (train_index, test_index) in kf.split(trainx):
        subtrainx, subvalidx = trainx[train_index,...], trainx[test_index,...]
        subtrainy, subvalidy = trainy[train_index], trainy[test_index]

        optuna_model.fit(subtrainx, subtrainy)
        y_pred_prob = optuna_model.predict_proba(subvalidx)[:,1]
        score = average_precision_score(subvalidy, y_pred_prob)
        results.append(score)

    return np.mean(results)
