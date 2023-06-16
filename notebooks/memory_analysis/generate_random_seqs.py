import argparse
import os
import sys
import time
import numpy as np
import xarray as xr
import seqpro as sp
import seqdata as sd

def random_seqs(
    shape,
    alphabet,
    rng = None,
    seed = None
):
    """Generate random nucleotide sequences.

    Parameters
    ----------
    shape : int, tuple[int]
        Shape of sequences to generate
    alphabet : NucleotideAlphabet
        Alphabet to sample nucleotides from.
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray
        Randomly generated sequences.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.choice(alphabet.array, size=shape)

def random_cov(
    shape,
    rate=1,
    rng = None,
    seed = None
):
    """Generate random coverage for nucleotide sequences.

    Parameters
    ----------
    shape : int, tuple[int]
        Shape of sequences to generate
    alphabet : NucleotideAlphabet
        Alphabet to sample nucleotides from.
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray
        Randomly generated sequences.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.poisson(rate, size=shape)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate random sequence datasets')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('--num_seqs', type=str, default='100,1000,10000,100000,1000000,10000000', help='Comma separated list of number of sequences to generate')
    parser.add_argument('--seq_lens', type=str, default='100, 1000, 10000, 100000, 1000000, 10000000', help='Comma separated list of sequence lengths to generate')
    parser.add_argument('--cov', action='store_true', help='Generate coverage data')
    parser.add_argument('--cov_dim', type=int, default=2, help='Coverage dimension')
    parser.add_argument('--seed', type=int, default=13, help='Random seed')
    return parser.parse_args()


def generate_datasets(output_dir, num_seqs, seq_lens, cov, cov_dim, seed):
    
    # Define an rng for reproducibility
    rng = np.random.default_rng(seed)

    # Generate the datasets
    for n in num_seqs:
        for l in seq_lens:

            # Generate the sequences
            shape = (n, l)
            print(f"Generating {n} sequences of length {l}")
            start_time = time.time()
            seqs = random_seqs(shape, alphabet=sp.alphabets.DNA, seed=seed)
            mem_usage = sys.getsizeof(seqs)
            mem_usage_gb = mem_usage / 1e9
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Generated {n} sequences of length {l} in {elapsed_time:.2f} seconds, using {mem_usage_gb} GBs of memory")

            # Generate the coverage data
            if cov:
                print(f"Generating {cov_dim} tracks of coverage data for {n} sequences of length {l}")
                start_time = time.time()
                covs = random_cov((n, cov_dim, l), rng=rng)
                mem_usage = sys.getsizeof(covs)
                mem_usage_gb = mem_usage / 1e9
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Generated {cov_dim} tracks of coverage data for {n} sequences of length {l} in {elapsed_time:.2f} seconds, using {mem_usage_gb} GBs of memory")

                # Save the dataset
                print(f"Saving {n} sequences of length {l}")
                start_time = time.time()
                sdata = xr.Dataset(
                    {
                        "seqs": (["seq", "pos"], seqs),
                        "cov": (["seq", "track", "pos"], covs),
                    }
                )
                sdata.to_zarr(f"{output_dir}/{n}_random_{l}bp_seqs_{cov_dim}d_cov.zarr", mode="w", overwrite=True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Saved {n} sequences of length {l} in {elapsed_time:.2f} seconds")

            else:
                # Save the dataset
                print(f"Saving {n} sequences of length {l}")
                start_time = time.time()
                sdata = xr.Dataset({"seqs": (["seq", "pos"], seqs)})
                sdata.to_zarr(f"{output_dir}/{n}_random_{l}bp_seqs.zarr", mode="w")
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Saved {n} sequences of length {l} in {elapsed_time:.2f} seconds")


def main():
    args = parse_args()
    num_seqs = [int(x) for x in args.num_seqs.split(',')]
    seq_lens = [int(x) for x in args.seq_lens.split(',')]
    if args.cov and not args.cov_dim:
        print('Error: cov_dim argument is required when generating coverage data')
        sys.exit(1)
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generate_datasets(output_dir, num_seqs, seq_lens, args.cov, args.cov_dim, args.seed)


if __name__ == '__main__':
    main()
