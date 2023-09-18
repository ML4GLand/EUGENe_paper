import numpy as np
np.random.seed(13)
from tqdm.auto import tqdm
from typing import List, Union, Optional, Iterable
import pandas as pd
from itertools import product
from typing import List
tqdm.pandas()
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from eugene.evaluate import median_calc, auc_calc, escore


# VOCABS -- for DNA/RNA only for now
DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
COMPLEMENT_DNA = {"A": "T", "C": "G", "G": "C", "T": "A"}
COMPLEMENT_RNA = {"A": "U", "C": "G", "G": "C", "U": "A"}


# exact concise
def _get_vocab(vocab):
    if vocab == "DNA":
        return DNA
    elif vocab == "RNA":
        return RNA
    else:
        raise ValueError("Invalid vocab, only DNA or RNA are currently supported")


# exact concise
def _get_vocab_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    Used in `_tokenize`.
    """
    return {l: i for i, l in enumerate(vocab)}


# exact concise
def _get_index_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    """
    return {i: l for i, l in enumerate(vocab)}


# modified concise
def _tokenize(seq, vocab="DNA", neutral_vocab=["N"]):
    """
    Convert sequence to integers based on a vocab
    Parameters
    ----------
    seq: 
        sequence to encode
    vocab: 
        vocabulary to use
    neutral_vocab: 
        neutral vocabulary -> assign those values to -1
    
    Returns
    -------
        List of length `len(seq)` with integers from `-1` to `len(vocab) - 1`
    """
    vocab = _get_vocab(vocab)
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]

    nchar = len(vocab[0])
    for l in vocab + neutral_vocab:
        assert len(l) == nchar
    assert len(seq) % nchar == 0  # since we are using striding

    vocab_dict = _get_vocab_dict(vocab)
    for l in neutral_vocab:
        vocab_dict[l] = -1

    # current performance bottleneck
    return [
        vocab_dict[seq[(i * nchar) : ((i + 1) * nchar)]]
        for i in range(len(seq) // nchar)
    ]


# my own
def _sequencize(tvec, vocab="DNA", neutral_value=-1, neutral_char="N"):
    """
    Converts a token vector into a sequence of symbols of a vocab.
    """
    vocab = _get_vocab(vocab) 
    index_dict = _get_index_dict(vocab)
    index_dict[neutral_value] = neutral_char
    return "".join([index_dict[i] for i in tvec])


# modified concise
def _token2one_hot(tvec, vocab="DNA", fill_value=None):
    """
    Converts an L-vector of integers in the range [0, D] into an L x D one-hot
    encoding. If fill_value is not None, then the one-hot encoding is filled
    with this value instead of 0.
    Parameters
    ----------
    tvec : np.array
        L-vector of integers in the range [0, D]
    vocab_size : int
        D
    fill_value : float, optional
        Value to fill the one-hot encoding with. If None, then the one-hot
    """
    vocab = _get_vocab(vocab)
    vocab_size = len(vocab)
    arr = np.zeros((vocab_size, len(tvec)))
    tvec_range = np.arange(len(tvec))
    tvec = np.asarray(tvec)
    arr[tvec[tvec >= 0], tvec_range[tvec >= 0]] = 1
    if fill_value is not None:
        arr[:, tvec_range[tvec < 0]] = fill_value
    return arr.astype(np.int8) if fill_value is None else arr.astype(np.float16)


# modified dinuc_shuffle
def _one_hot2token(one_hot, neutral_value=-1, consensus=False):
    """
    Converts a one-hot encoding into a vector of integers in the range [0, D]
    where D is the number of classes in the one-hot encoding.
    Parameters
    ----------
    one_hot : np.array
        L x D one-hot encoding
    neutral_value : int, optional
        Value to use for neutral values.
    
    Returns
    -------
    np.array
        L-vector of integers in the range [0, D]
    """
    if consensus:
        return np.argmax(one_hot, axis=0)
    tokens = np.tile(neutral_value, one_hot.shape[1])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot.transpose()==1)
    tokens[seq_inds] = dim_inds
    return tokens


# pad and subset, exact concise
def _pad(seq, max_seq_len, value="N", align="end"):
    seq_len = len(seq)
    assert max_seq_len >= seq_len
    if align == "end":
        n_left = max_seq_len - seq_len
        n_right = 0
    elif align == "start":
        n_right = max_seq_len - seq_len
        n_left = 0
    elif align == "center":
        n_left = (max_seq_len - seq_len) // 2 + (max_seq_len - seq_len) % 2
        n_right = (max_seq_len - seq_len) // 2
    else:
        raise ValueError("align can be of: end, start or center")

    # normalize for the length
    n_left = n_left // len(value)
    n_right = n_right // len(value)

    return value * n_left + seq + value * n_right

# exact concise
def _trim(seq, maxlen, align="end"):
    seq_len = len(seq)

    assert maxlen <= seq_len
    if align == "end":
        return seq[-maxlen:]
    elif align == "start":
        return seq[0:maxlen]
    elif align == "center":
        dl = seq_len - maxlen
        n_left = dl // 2 + dl % 2
        n_right = seq_len - dl // 2
        return seq[n_left:n_right]
    else:
        raise ValueError("align can be of: end, start or center")


# modified concise
def _pad_sequences(
    seqs, 
    maxlen=None, 
    align="end", 
    value="N"
):
    """
    Pads sequences to the same length.
    Parameters
    ----------
    seqs : list of str
        Sequences to pad
    maxlen : int, optional
        Length to pad to. If None, then pad to the length of the longest sequence.
    align : str, optional
        Alignment of the sequences. One of "start", "end", "center"
    value : str, optional
        Value to pad with
    Returns
    -------
    np.array
        Array of padded sequences
    """

    # neutral element type checking
    assert isinstance(value, list) or isinstance(value, str)
    assert isinstance(value, type(seqs[0])) or type(seqs[0]) is np.str_
    assert not isinstance(seqs, str)
    assert isinstance(seqs[0], list) or isinstance(seqs[0], str)

    max_seq_len = max([len(seq) for seq in seqs])

    if maxlen is None:
        maxlen = max_seq_len
    else:
        maxlen = int(maxlen)

    if max_seq_len < maxlen:
        import warnings
        warnings.warn(
            f"Maximum sequence length ({max_seq_len}) is smaller than maxlen ({maxlen})."
        )
        max_seq_len = maxlen

    # check the case when len > 1
    for seq in seqs:
        if not len(seq) % len(value) == 0:
            raise ValueError("All sequences need to be dividable by len(value)")
    if not maxlen % len(value) == 0:
        raise ValueError("maxlen needs to be dividable by len(value)")

    padded_seqs = [
        _trim(_pad(seq, max(max_seq_len, maxlen), value=value, align=align), maxlen, align=align)
        for seq in seqs 
    ]
    return padded_seqs


# modifed concise
def ohe_seq(
    seq: str, 
    vocab: str = "DNA", 
    neutral_vocab: str = "N", 
    fill_value: int = 0
) -> np.array:
    """Convert a sequence into one-hot-encoded array."""
    seq = seq.strip().upper()
    return _token2one_hot(_tokenize(seq, vocab, neutral_vocab), vocab, fill_value=fill_value)


# modfied concise
def ohe_seqs(
    seqs: Iterable[str],
    vocab: str = "DNA",
    neutral_vocab: Union[str, List[str]] = "N",
    maxlen: Optional[int] = None,
    pad: bool = True,
    pad_value: str = "N",
    fill_value: Optional[str] = None,
    seq_align: str = "start",
    verbose: bool = True,
) -> np.ndarray:
    """Convert a set of sequences into one-hot-encoded array."""
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]
    if isinstance(seqs, str):
        raise ValueError("seq_vec should be an iterable not a string itself")
    assert len(vocab[0]) == len(pad_value)
    assert pad_value in neutral_vocab
    if pad:
        seqs_vec = _pad_sequences(seqs, maxlen=maxlen, align=seq_align, value=pad_value)
    arr_list = [
        ohe_seq(
            seq=seqs_vec[i],
            vocab=vocab,
            neutral_vocab=neutral_vocab,
            fill_value=fill_value,
        )
        for i in tqdm(
            range(len(seqs_vec)),
            total=len(seqs_vec),
            desc="One-hot encoding sequences",
            disable=not verbose,
        )
    ]
    if pad:
        return np.stack(arr_list)
    else:
        return np.array(arr_list, dtype=object)


# Useful helpers for generating and checking for kmers
def generate_all_possible_kmers(n=7, alphabet="AGCU"):
    """Generate all possible kmers of length and alphabet provided
    """
    return ["".join(c) for c in product(alphabet, repeat=n)]


def kmer_in_seqs(seqs, kmer):
    """Return a 0/1 array of whether a kmer is in each of the passed in sequences
    """
    seqs_s = pd.Series(seqs)
    kmer_binary = seqs_s.str.contains(kmer).astype(int).values
    return kmer_binary


def rnacomplete_metrics(
    kmer_presence_mtx: np.ndarray, 
    intensities: np.ndarray, 
    verbose: bool = True, 
    swifter: bool = False
):
    """
    Calculate the RNAcomplete metrics for a set of k-mers and scores for a set of sequences.

    Parameters
    ----------
    kmer_presence_mtx : np.ndarray
        A binary matrix of k-mers x samples. A 1 in entry (i, j) indicates that sequence j contains k-mer i. 
    intensities : np.ndarray
        A vector of scores for each sequence.
    """
    y_score = intensities
    df = pd.DataFrame(kmer_presence_mtx).astype(np.int8)
    if verbose:
        if not swifter:
            rbp_eval = df.progress_apply(
                lambda y_true: pd.Series(
                    {
                        "Median": median_calc(y_true, y_score),
                        "AUC": auc_calc(y_true, y_score),
                        "E-score": escore(y_true, y_score),
                    }
                ),
                axis=1,
            )
        else:
            try:
                import swifter
            except ImportError:
                raise ImportError(
                    "swifter is not installed. Please install swifter to use this feature."
                )
            rbp_eval = df.swifter.apply(
                lambda y_true: pd.Series(
                    {
                        "Median": median_calc(y_true, y_score),
                        "AUC": auc_calc(y_true, y_score),
                        "E-score": escore(y_true, y_score),
                    }
                ),
                axis=1,
            )

    else:
        rbp_eval = df.apply(
            lambda y_true: pd.Series(
                {
                    "Median": median_calc(y_true, y_score),
                    "AUC": auc_calc(y_true, y_score),
                    "E-score": escore(y_true, y_score),
                }
            ),
            axis=1,
        )
    rbp_eval["Z-score"] = (rbp_eval["Median"] - np.mean(rbp_eval["Median"])) / np.std(
        rbp_eval["Median"], ddof=1
    )
    return (
        rbp_eval["Z-score"].values,
        rbp_eval["AUC"].values,
        rbp_eval["E-score"].values,
    )


def rnacomplete_metrics_sdata_plot(
    sdata,
    kmer_presence_mtx: np.ndarray,
    target_var: str,
    return_cors: bool = False,
    verbose: bool = True,
    preds_suffix: str = "_predictions",
    **kwargs
):
    """
    Calculate the RNAcomplete metrics for a set of k-mers and intensities in a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        A SeqData object containing the intensities for each sequence in seqs_annot.
    kmer_presence_mtx : np.ndarray
        A binary matrix of k-mers x samples. A 1 in entry (i, j) indicates that sequence j contains k-mer i.
    target_var : str
        The key in sdata.seqs_annot to use for the intensities.
    return_cors : bool, optional
        Whether to return the Pearson and Spearman correlations, by default False (plots only) 
    verbose : bool, optional
        Whether to show a progress bar for all the k-mers, by default True
    preds_suffix : str, optional
        The suffix to use for the predictions, by default "_predictions"
    """
    observed = sdata[target_var].values
    preds = sdata[f"{target_var}{preds_suffix}"].values
    
    # Get zscores, aucs and escores from observed intensities
    observed_zscores, observed_aucs, observed_escores = rnacomplete_metrics(
        kmer_presence_mtx, observed, verbose=verbose, **kwargs
    )

    # Get zscores, aucs, and escores from predicted intensities
    preds_zscores, preds_aucs, preds_escores = rnacomplete_metrics(
        kmer_presence_mtx, preds, verbose=verbose, **kwargs
    )
    # Z-scores
    zscore_nan_mask = np.isnan(observed_zscores) | np.isnan(preds_zscores)
    preds_zscores = preds_zscores[~zscore_nan_mask]
    observed_zscores = observed_zscores[~zscore_nan_mask]
    if len(observed_zscores) > 0 and len(preds_zscores) > 0:
        zscore_pearson = pearsonr(preds_zscores, observed_zscores)[0]
        zscore_spearman = spearmanr(preds_zscores, observed_zscores).correlation
    else:
        zscore_pearson = np.nan
        zscore_spearman = np.nan

    # AUCs
    auc_nan_mask = np.isnan(observed_aucs) | np.isnan(preds_aucs)
    preds_aucs = preds_aucs[~auc_nan_mask]
    observed_aucs = observed_aucs[~auc_nan_mask]
    auc_pearson = pearsonr(preds_aucs, observed_aucs)[0]
    auc_spearman = spearmanr(preds_aucs, observed_aucs).correlation

    # E-scores
    escore_nan_mask = np.isnan(observed_escores) | np.isnan(preds_escores)
    preds_escores = preds_escores[~escore_nan_mask]
    observed_escores = observed_escores[~escore_nan_mask]
    escore_pearson = pearsonr(preds_escores, observed_escores)[0]
    escore_spearman = spearmanr(preds_escores, observed_escores).correlation

    # Intensities
    intensity_nan_mask = np.isnan(observed) | np.isnan(preds)
    preds = preds[~intensity_nan_mask]
    observed = observed[~intensity_nan_mask]
    intensity_pearson = pearsonr(observed, preds)[0]
    intensity_spearman = spearmanr(observed, preds).correlation

    if return_cors:
        pearson = {
            "Z-score": zscore_pearson,
            "AUC": auc_pearson,
            "E-score": escore_pearson,
            "Intensity": intensity_pearson,
        }
        spearman = {
            "Z-score": zscore_spearman,
            "AUC": auc_spearman,
            "E-score": escore_spearman,
            "Intensity": intensity_spearman,
        }
        return pearson, spearman

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].scatter(observed_zscores, preds_zscores)
    ax[0].set_title("Z-scores")
    ax[0].set_xlabel("Observed")
    ax[0].set_ylabel("Predicted")
    ax[0].text(
        0.75,
        0.05,
        "r="
        + str(round(zscore_pearson, 2))
        + "\nrho="
        + str(round(zscore_spearman, 2)),
        transform=ax[0].transAxes,
    )

    ax[1].scatter(observed_aucs, preds_aucs)
    ax[1].set_title("AUCs")
    ax[1].set_xlabel("Observed")
    ax[1].set_ylabel("Predicted")
    ax[1].text(
        0.75,
        0.05,
        "r=" + str(round(auc_pearson, 2)) + "\nrho=" + str(round(auc_spearman, 2)),
        transform=ax[1].transAxes,
    )

    ax[2].scatter(observed_escores, preds_escores)
    ax[2].set_title("E-scores")
    ax[2].set_xlabel("Observed")
    ax[2].set_ylabel("Predicted")
    ax[2].text(
        0.75,
        0.05,
        "r="
        + str(round(escore_pearson, 2))
        + "\nrho="
        + str(round(escore_spearman, 2)),
        transform=ax[2].transAxes,
    )

    ax[3].scatter(observed, preds)
    ax[3].set_title("Intensities")
    ax[3].set_xlabel("Observed")
    ax[3].set_ylabel("Predicted")
    ax[3].text(
        0.75,
        0.05,
        "r="
        + str(round(intensity_pearson, 2))
        + "\nrho="
        + str(round(intensity_spearman, 2)),
        transform=ax[3].transAxes,
    )

    plt.tight_layout()


def rnacomplete_metrics_sdata_table(
    sdata,
    kmer_presence_mtx: np.ndarray,
    target_vars: List[str],
    num_kmers: int = 100,
    verbose: bool = False,
    preds_suffix: str = "_predictions",
    **kwargs
):
    """
    Generate a table of RNAcomplete metrics for a list of target keys.

    Parameters
    ----------
    sdata : SeqData
        SeqData object with observed and predicted scores in columns of seqs_annot
    kmer_presence_mtx : np.ndarray
       A binary matrix of k-mers x samples. A 1 in entry (i, j) indicates that sequence j contains k-mer i.
    target_vars : List[str]
        List of target keys to compute metrics for
    num_kmers : int, optional
        Number of k-mers to sample to compute metrics, by default 100. For large sets of k-mers this can take long
    verbose : bool, optional
        Whether to print progress, by default False
    preds_suffix : str, optional
        Suffix of predicted scores in seqs_annot, by default "_predictions"
    
    Returns
    -------
    pd.DataFrame
        A table of RNAcomplete metrics for each target key
    """
    if isinstance(target_vars, str):
        target_vars = [target_vars]
    spearman_summary = pd.DataFrame()
    pearson_summary = pd.DataFrame()
    if num_kmers is not None:
        random_kmers = np.random.choice(np.arange(kmer_presence_mtx.shape[0]), size=num_kmers)
        kmer_presence_mtx = kmer_presence_mtx[random_kmers, :]
    valid_kmers = np.where(np.sum(kmer_presence_mtx, axis=1) > 155)[0]
    kmer_presence_mtx = kmer_presence_mtx[valid_kmers, :]
    print(kmer_presence_mtx.shape)
    for i, target_var in tqdm(
        enumerate(target_vars), desc="Evaluating probes", total=len(target_vars)
    ):
        rs, rhos = rnacomplete_metrics_sdata_plot(
            sdata,
            kmer_presence_mtx,
            target_var=target_var,
            return_cors=True,
            verbose=verbose,
            preds_suffix=preds_suffix,
            **kwargs,
        )
        pearson_summary = pd.concat(
            [pearson_summary, pd.DataFrame(rs, index=[target_var])], axis=0
        )
        spearman_summary = pd.concat(
            [spearman_summary, pd.DataFrame(rhos, index=[target_var])], axis=0
        )
    return pearson_summary, spearman_summary
