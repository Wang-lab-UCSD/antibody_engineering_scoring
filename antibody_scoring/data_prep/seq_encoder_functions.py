"""Contains tools for encoding the input sequences."""
import numpy as np
from xGPR import KernelFGen

from ..constants import seq_encoding_constants
from ..protein_tools.pfasum_matrices import PFASUM90_standardized




class OneHotEncoder():
    """Generates one-hot encoding for sequences."""

    def __init__(self):
        self.aa_dict = {seq_encoding_constants.aas[i]:i for i
                in range(len(seq_encoding_constants.aas))}

    def encode(self, sequences):
        """Encodes the input sequences as a 3d array."""
        encoded_sequences = []
        for sequence in sequences:
            encoded_seq = np.zeros((len(sequence), 21), dtype = np.uint8)
            for i, letter in enumerate(list(sequence)):
                encoded_seq[i, self.aa_dict[letter]] = 1
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences


    def encode_variable_length(self, sequences):
        """Encodes the input sequences as a 3d array when
        sequences are variable in length."""
        encoded_sequences = []
        max_len = max([len(seq) for seq in sequences])

        for sequence in sequences:
            encoded_seq = np.zeros((max_len, 21), dtype = np.uint8)
            for i, letter in enumerate(list(sequence)):
                encoded_seq[i, self.aa_dict[letter]] = 1
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences




class PFAStandardEncoder():
    """Encodes a sequence using a PFASUM (similar to BLOSUM) matrix,
    without inserting any gaps. The matrix used here has been
    standardized and undergone a Cholesky decomposition."""
    def __init__(self):
        aas = PFASUM90_standardized.get_aas()
        self.aa_dict = {aas[i]:i for i
                in range(len(aas))}
        self.mat = PFASUM90_standardized.get_mat()

    def encode(self, sequences):
        """Encodes the input sequences as a 3d array."""
        encoded_sequences = []
        for sequence in sequences:
            encoded_seq = np.stack([self.mat[self.aa_dict[letter],:] for letter in sequence])
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences


    def encode_variable_length(self, sequences):
        """Encodes the input sequences as a 3d array when
        the input sequences have different lengths."""
        encoded_sequences = []
        max_len = max([len(seq) for seq in sequences])

        for sequence in sequences:
            encoded_seq = np.zeros((max_len, 21), dtype = np.float32)
            for i, letter in enumerate(list(sequence)):
                encoded_seq[i,:] = self.mat[self.aa_dict[letter],:]
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences



class PChemPropEncoder():
    """Encodes a sequence using four physicochemical properties (the
    properties from Grantham et al. and expected charge at neutral pH).
    These are a little arbitrary, but it's hard to come up with a list
    of pchem predictors that ISN'T a little arbitrary, and for deep
    learning (and even to some extent for simpler models) the specific
    descriptors do not matter terribly as long as there is sufficient
    info to differentiate AAs."""

    def __init__(self):
        self.compositionality = seq_encoding_constants.COMPOSITION
        self.volume = seq_encoding_constants.VOLUME
        self.polarity = seq_encoding_constants.POLARITY
        self.mat = {aa:np.array([self.compositionality[aa], self.volume[aa],
                                 self.polarity[aa]]) for aa in seq_encoding_constants.aas}


    def encode_variable_length(self, sequences):
        """Encodes the input sequences as a 3d array."""
        encoded_sequences = []
        max_len = max([len(seq) for seq in sequences])

        for sequence in sequences:
            features = np.zeros((max_len, 3), dtype = np.float32)
            for i, aa in enumerate(sequence):
                features[i,0] = self.compositionality[aa]
                features[i,1] = self.volume[aa]
                features[i,2] = self.polarity[aa]
            encoded_sequences.append(features)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences
