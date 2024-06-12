"""Contains code for evaluating xgboost and xgpr on the protein gym
dms substitutions dataset."""
import os
import time
import optuna
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from xGPR import xGPRegression, build_regression_dataset
import numpy as np
from ..data_prep.protein_gym_data_processing import preprocess_dms_substitutions
from ..data_prep.protein_gym_data_processing import extract_tt_splits
from ..data_prep.seq_encoder_functions import OneHotEncoder, PFAStandardEncoder, PChemPropEncoder
from .shared_eval_funcs import optuna_regression


def write_prot_gym_logfile(project_dir, filename, model_type, encoding,
        split_type, spearman_scores, fit_times):
    """Writes the result of an evaluation to file. This is much simpler for
    protein gym since we only need to record spearman's-r."""
    base_fname = os.path.basename(filename)
    assay_id = base_fname.split(".csv")[0]
    if "dms_indels" in filename:
        dataset_type = "dms_indels"
    elif "dms_subs" in filename:
        dataset_type = "dms_subs"
    else:
        raise RuntimeError("Unexpected dataset type encountered.")

    output_fname = os.path.join(project_dir, "results", "protein_gym_results.txt")

    if not os.path.exists(output_fname):
        with open(output_fname, "w+", encoding="utf-8") as fhandle:
            fhandle.write("Assay_ID,Dataset_Type,Split_Type,Model,Encoding,"
                "Spearman_fitness,Std_dev_Spearman_fitness,Average_fit_time\n")

    with open(output_fname, "a", encoding="utf-8") as fhandle:
        fhandle.write(f"{assay_id},{dataset_type},{split_type},{model_type},{encoding},"
                f"{np.mean(spearman_scores)},{np.std(spearman_scores)},"
                f"{np.mean(fit_times)}\n")



def dms_subs_eval(project_dir):
    """Runs the evals for the protein gym dms subs dataset for both
    xgpr and xgboost."""
    i = 0
    for sequences, yvalues, filename in preprocess_dms_substitutions(project_dir):
        if i > 0:
            xgpr_eval(project_dir, sequences, yvalues, OneHotEncoder(),
                    filename)
            xgpr_eval(project_dir, sequences, yvalues, OneHotEncoder(),
                filename, kernel_choice = "Conv1dRBF")
        i += 1
        if i > 1:
            break




def xgpr_eval(project_dir, sequences, yvalues,
        encoder, filename, split_type = "random",
        kernel_choice = "RBF"):
    """Runs train-test evaluations using a specified train fraction."""
    spearman_scores, fit_times = [], []

    sequence_batches, yvalue_batches = extract_tt_splits(123, sequences,
            yvalues, split_type)

    conv_kernel = kernel_choice.startswith("Conv")
    use_temp_files = len(sequences) > 1e4

    for i, (testseqs, testyvals) in enumerate(zip(sequence_batches, yvalue_batches)):
        testx, testy, testl = prep_seq_data(project_dir, testseqs, testyvals,
                encoder, use_temp_files, conv_kernel, prefix="test")

        trainseqs, trainyvals = [], []
        for j in range(5):
            if j != i:
                trainseqs += sequence_batches[j]
                trainyvals += yvalue_batches[j]

        trainx, trainy, trainl = prep_seq_data(project_dir, trainseqs, trainyvals,
                encoder, use_temp_files, conv_kernel, prefix="train")

        timestamp = time.time()
        regdata = build_regression_dataset(trainx, trainy, sequence_lengths = trainl, chunk_size=2000)
        xgp = xGPRegression(num_rffs = 2048, variance_rffs = 12, kernel_choice = kernel_choice,
                kernel_settings = {"intercept":True, "conv_width":9,
                    "averaging":"sqrt"}, device="gpu")

        _ = xgp.tune_hyperparams_crude(regdata)
        xgp.num_rffs = 4096

        xgp.tune_hyperparams(regdata)
        xgp.num_rffs = 16384
        xgp.fit(regdata, suppress_var = True)
        fit_times.append(time.time() - timestamp)

        if use_temp_files:
            if conv_kernel:
                preds = np.concatenate([xgp.predict(np.load(x), np.load(l)) for (x,l) in
                    zip(testx, testl)  ])
            else:
                preds = np.concatenate([xgp.predict(np.load(x)) for x in testx])

        else:
            preds = xgp.predict(testx, sequence_lengths=testl)

        if use_temp_files:
            test_gt = np.concatenate([np.load(y) for y in testy])
        else:
            test_gt = testy

        spearman_scores.append(spearmanr(preds, test_gt)[0])

    if use_temp_files:
        cleanup_files(trainx, trainy, trainl)
        cleanup_files(testx, testy, testl)

    write_prot_gym_logfile(project_dir, filename, f"xGPR_{kernel_choice}", type(encoder).__name__,
            split_type, spearman_scores, fit_times)



def prep_seq_data(project_dir, sequence_list, yvalue_list,
        encoder, use_temp_files = False, conv_kernel = False,
        prefix = "test"):
    """Decides whether to save the data to file or convert it to arrays
    and returns the appropriately formatted arrays or file lists."""
    if use_temp_files:
        xdata, ydata, ldata = save_chunked_data(project_dir, sequence_list, yvalue_list,
                    encoder, 2000, conv_kernel, prefix)
    else:
        xdata = encoder.encode_variable_length(sequence_list).astype(np.float32)

        if conv_kernel:
            ldata = np.array([len(s) for s in sequence_list], dtype=np.int32)
        else:
            ldata = None
            xdata = flatten_arr(xdata)

        ydata = np.array(yvalue_list)

    return xdata, ydata, ldata



def save_chunked_data(project_dir, sequence_list, yvalue_list,
        encoder, chunk_size = 2000, conv_kernel = False,
        prefix = "test"):
    """Saves data to file in chunks so that model fitter can iterate over
    them."""
    out_dir = os.path.join(project_dir, "temp")
    os.makedirs(out_dir, exist_ok=True)

    xfiles, yfiles, lfiles = [], [], []

    for i in range(0, len(sequence_list), chunk_size):
        xdata = encoder.encode_variable_length(sequence_list[i:i+chunk_size])
        if not conv_kernel:
            xdata = flatten_arr(xdata)
        ydata = np.array(yvalue_list[i:i+chunk_size])

        xfiles.append(os.path.join(out_dir, f"{prefix}_{i}_xdata.npy"))
        yfiles.append(os.path.join(out_dir, f"{prefix}_{i}_ydata.npy"))

        np.save(xfiles[-1], xdata)
        np.save(yfiles[-1], ydata)

        if conv_kernel:
            lfiles.append(os.path.join(out_dir, f"{prefix}_{i}_ldata.npy"))
            ldata = np.array([len(s) for s in sequence_list[i:i+chunk_size]],
                dtype=np.int32)
            np.save(lfiles[-1], ldata)

    if not conv_kernel:
        lfiles = None

    return xfiles, yfiles, lfiles


def cleanup_files(xfiles, yfiles, lfiles):
    """Convenience function that removes all the temporary files created to
    store data."""
    if lfiles is None:
        for xfile, yfile in zip(xfiles, yfiles):
            os.remove(xfile)
            os.remove(yfile)
        return
    for xfile, yfile, lfile in zip(xfiles, yfiles, lfiles):
        os.remove(xfile)
        os.remove(yfile)
        os.remove(lfile)


def flatten_arr(input_array):
    """Convenience function for flattening an array."""
    return input_array.reshape((input_array.shape[0], input_array.shape[1] *
            input_array.shape[2]))
