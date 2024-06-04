"""Contains shared functions for hyperparameter tuning and result handling shared by
all of the eval modules."""
import os
import numpy as np
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn.model_selection import KFold


def write_res_to_file(project_dir, dataset_name, model_type, encoding,
        r2_scores = [], mae_scores = [], auc_roc_scores = [],
        auc_prc_scores = [], fit_times = []):
    """Writes the result of an evaluation to file."""
    output_fname = os.path.join(project_dir, "results", "evaluation_results.txt")
    if not os.path.exists(output_fname):
        with open(output_fname, "w+", encoding="utf-8") as fhandle:
            fhandle.write("Dataset,Model,Encoding,R^2,MAE,AUC_ROC,AUC_PRC,Fit_time\n")

    with open(output_fname, "a", encoding="utf-8") as fhandle:
        fhandle.write(f"{dataset_name},{model_type},{encoding},"
                f"{convert_scores_to_stats(r2_scores)},"
                f"{convert_scores_to_stats(mae_scores)},"
                f"{convert_scores_to_stats(auc_roc_scores)},"
                f"{convert_scores_to_stats(auc_prc_scores)},"
                f"{convert_scores_to_stats(fit_times)}\n")



def convert_scores_to_stats(scores):
    """Takes a list of scores and if empty returns NA, otherwise
    converts it to a mean and 95% CI."""
    if len(scores) == 0:
        return "NA"
    conf_int_h = norm.interval(0.95, loc=0,
            scale=np.std(scores, ddof=1) / np.sqrt(len(scores)))[1]
    return f"{np.mean(scores)}+/-{conf_int_h}"




def optuna_regression(trial, trainx, trainy):
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

    optuna_model = XGBRegressor(**params)
    results = []
    for (train_index, test_index) in kf.split(trainx):
        subtrainx, subvalidx = trainx[train_index,...], trainx[test_index,...]
        subtrainy, subvalidy = trainy[train_index], trainy[test_index]

        optuna_model.fit(subtrainx, subtrainy)
        y_pred = optuna_model.predict(subvalidx)
        score = np.mean(np.abs(subvalidy - y_pred))
        results.append(score)

    return np.mean(results)
