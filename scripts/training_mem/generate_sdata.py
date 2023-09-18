from pathlib import Path
from typing import Annotated

import typer


def main(
    n_seqs: int,
    seq_length: int,
    output: Path,
    max_memory: Annotated[int, typer.Argument(help="Max memory in mebibytes.")],
):
    import logging
    import sys
    from time import perf_counter

    import numcodecs as ncds
    import numpy as np
    import zarr
    from tqdm import tqdm

    logger = logging.getLogger("GENERATE_SDATA")
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info(f"Generating data: {n_seqs=} {seq_length=}")

    # 1024**2 bytes per mebibyte
    max_batch_size = (max_memory * 1024**2) // (16 * seq_length)
    logger.info(
        f"Setting max batch size to {max_batch_size} based on memory constraints."
    )

    alphabet = np.frombuffer(b"ACGT", "S1")

    store = zarr.open_group(output, mode="w")
    store.attrs["sequence_dim"] = "_sequence"
    store.attrs["length_dim"] = "_length"
    store.attrs["max_jitter"] = 0

    compressor = ncds.Blosc("zstd", clevel=7, shuffle=-1)

    seq = store.create(
        "seq",
        shape=(n_seqs, seq_length),
        chunks=(min(max_batch_size, n_seqs), None),
        dtype=alphabet.dtype,
        compressor=compressor,
    )
    seq.attrs["_ARRAY_DIMENSIONS"] = ["_sequence", "_length"]

    logger.info("Adding sequences to Zarr store.")

    rng = np.random.default_rng()
    pbar = tqdm(total=n_seqs, unit="sequence")
    t0 = perf_counter()
    times = []
    for i in range(0, n_seqs, max_batch_size):
        # don't go past n_seqs
        batch_size = min(max_batch_size, n_seqs - i)
        t1 = perf_counter()
        seq[i : i + batch_size] = rng.choice(alphabet, (batch_size, seq_length))
        times.append(perf_counter() - t1)
        pbar.update(batch_size)

    logger.info(f"Generated and wrote data in {perf_counter() - t0:.2f} seconds.")
    logger.info(f"Median time to write a batch: {np.median(times):.4f} seconds.")

    zarr.consolidate_metadata(output)  # type: ignore


if __name__ == "__main__":
    typer.run(main)
