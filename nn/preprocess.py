# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # set up positive and negative sequences
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label]
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]
    
    # calculate number of sequences to sample from each class
    n_pos = len(pos_seqs)
    n_neg = len(neg_seqs)
    n_samples = min(n_pos, n_neg)
    
    # sample the sequences
    pos_samples = np.random.choice(pos_seqs, n_samples, replace=True)
    neg_samples = np.random.choice(neg_seqs, n_samples, replace=True)
    
    # Combine the sampled sequences and labels
    sampled_seqs = list(pos_samples) + list(neg_samples)
    sampled_labels = [True] * n_samples + [False] * n_samples
    
    # shuffle the sequences and labels
    shuffle_idx = np.random.permutation(len(sampled_seqs))
    sampled_seqs = [sampled_seqs[i] for i in shuffle_idx]
    sampled_labels = [sampled_labels[i] for i in shuffle_idx]

    # return sampled sequences and their lables
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # one-hot encoding dictionary
    encoding_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}

    encodings = []

    # for each sequence
    for seq in seq_arr:
        #store the one-hot encoding of this sequence
        encoding = []
        # one hot encode each nucleotide
        for nt in seq:
            encoding += encoding_dict[nt]
        encodings.append(encoding)
    return np.array(encodings)
    
def process_negative_sequences(neg_seqs: List[str], target_length: int) -> List[str]:
    # ensure that postive and negative sequences are the same length by splitting into subsequences

    processed_seqs = []
    for seq in neg_seqs:
        for i in range(len(seq) - target_length + 1):
            sub_seq = seq[i:i + target_length]
            processed_seqs.append(sub_seq)
    return processed_seqs

def train_test_split_nn(X: ArrayLike, y: ArrayLike, test_size: float = 0.2, random_state: int = None) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:

    # train test split function
    
    if random_state is not None:
        np.random.seed(random_state)

    # split based on number of samples provided in X
    n_samples = len(X)
    shuffle_idx = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))

    X_train = X[shuffle_idx[:split_idx]]
    X_val = X[shuffle_idx[split_idx:]]
    y_train = y[shuffle_idx[:split_idx]]
    y_val = y[shuffle_idx[split_idx:]]

    return X_train, X_val, y_train, y_val