"""Contains code for evaluating xgboost and xgpr on the Engelhart dataset."""
import numpy as np
from .data_processing import preprocess_engelhart
from .seq_encoder_functions import OneHotEncoder, PFAStandardEncoder, PChemPropEncoder



def engelhart_eval(project_dir):
    """Runs the evals for the Engelhart dataset for both xgpr and xgboost."""
    fixed_len_seqs, yvalues, cdr_unaligned, slengths = preprocess_engelhart(project_dir)

    barton_xgpr_conv_kernel_eval(cdr_unaligned, slengths, yvalues, OneHotEncoder)
    barton_xgpr_conv_kernel_eval(cdr_unaligned, slengths, yvalues, PFAStandardEncoder)
    barton_xgboost_eval(fixed_len_seqs, yvalues, PChemPropEncoder)


def barton_xgpr_conv_kernel_eval(cdr_unaligned, slengths, yvalues, encoder):
    """Runs train-test split evaluations using a split ratio specific
    to Barton et al. (to directly compare with that paper)."""
    pass

def barton_xgboost_eval(fixed_len_seqs, yvalues, encoder):
    """Runs train-test split evaluations using a split ratio
    specific to Barton et al. (to directly compare with that paper)."""
    pass



def write_res_to_file(
