"""Contains tools for encoding the input sequences."""
import time
import numpy as np
import ablang
from antpack import SingleChainAnnotator
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

class IntegerEncoder():
    """Encodes amino acid sequences as integers -- very lightweight
    and enables us to work with large datasets in memory. Should only
    be used with LightGBM and with sequences that are all the same
    length."""

    def __init__(self):
        self.aa_dict = {seq_encoding_constants.aas[i]:i for i
                in range(len(seq_encoding_constants.aas))}

    def encode(self, sequences):
        """Encodes the input sequences as a 2d array."""
        encoded_sequences = []
        for sequence in sequences:
            encoded_seq = np.array([self.aa_dict[letter] for letter in sequence],
                                   dtype=np.uint8)
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


class AbLangEncoder():
    """Encodes antibody sequences using AbLang."""

    def __init__(self):
        self.heavy_ablang = ablang.pretrained("heavy", device="cuda")
        self.heavy_ablang.freeze()
        self.light_ablang = ablang.pretrained("light", device="cuda")
        self.light_ablang.freeze()
        self.light_num_tool = SingleChainAnnotator(chains=["K", "L"], scheme="imgt")
        self.heavy_num_tool = SingleChainAnnotator(chains=["H"], scheme="imgt")


    def encode_variable_length(self, sequences, seq_type = "heavy"):
        """Returns the averaged reps corresponding to the sequence."""
        if seq_type == "heavy":
            rescodings = self.gen_codings(sequences, self.heavy_ablang)
        elif seq_type == "light":
            rescodings = self.gen_codings(sequences, self.light_ablang)

        elif seq_type == "both":
            light_seqs, heavy_seqs = [], []
            for seq in sequences:
                light_anal = self.light_num_tool.analyze_seq(seq)
                heavy_anal = self.heavy_num_tool.analyze_seq(seq)
                if light_anal[1] < 0.8 or heavy_anal[1] < 0.8:
                    raise RuntimeError("Expected both heavy and light chains in each "
                            "sequence but found a sequence that may have only one of the "
                            "two.")
                light_seqs.append(self.get_var_region(seq, light_anal))
                heavy_seqs.append(self.get_var_region(seq, heavy_anal))

            rescodings = self.gen_codings(heavy_seqs, self.heavy_ablang)
            rescodings_l = self.gen_codings(light_seqs, self.light_ablang)
            rescodings = np.hstack([rescodings, rescodings_l])

        else:
            raise RuntimeError("Unexpected sequence type supplied to sequence encoder.")

        return rescodings


    def gen_codings(self, sequences, model):
        """Generates the encodings in batches to avoid any possible memory
        issues."""
        codings = []
        for i in range(0, len(sequences), 250):
            codings.append(model(sequences[i:i+250], mode="seqcoding"))

        return np.vstack(codings)


    def get_var_region(self, sequence, seq_anal):
        """Returns the variable region based on the supplied numbering."""
        for i, s in enumerate(seq_anal[0]):
            if s != "-":
                break

        start_pt = i

        for i, s in enumerate(reversed(seq_anal[0])):
            if s != "-":
                break

        if i == 0:
            end_pt = len(seq_anal[0])
        else:
            end_pt = -i

        return sequence[start_pt:end_pt]


    def encode_mason_data(self, sequences):
        """The Mason dataset is a little 'special' in the sense
        that only the CDR is supplied. To generate an embedding
        we need to tack the rest of the sequence at the front and
        back back onto the CDR, then remove the parts of the embedding
        that do not correspond to the CDR."""
        front_add = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR"
        rear_add = "YWGQGTLVTVSS"

        aug_sequences = ["".join([front_add, s, rear_add]) for s
                         in sequences]

        codings = []
        for i in range(0, len(sequences), 250):
            coding = self.heavy_ablang(aug_sequences[i:i+250], mode="rescoding")
            coding = np.stack(coding)
            time.sleep(0.01)
            coding = coding[:,98:108,:].mean(axis=1)
            codings.append(coding)

        return np.vstack(codings)
