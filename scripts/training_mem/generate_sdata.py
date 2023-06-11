from pathlib import Path
from typing import Annotated

import typer


def main(
    n_seqs: int,
    seq_length: int,
    output: Path,
    max_memory: Annotated[
        str, typer.Argument(help="Max memory e.g. 32G, defaults to megabytes.")
    ],
):
    import logging
    import sys

    import numcodecs as ncds
    import numpy as np
    import zarr
    from tqdm import tqdm

    logger = logging.getLogger("GENERATE_SDATA")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Parsing max memory.")
    units = {"G": int(1e9), "M": int(1e6), "K": int(1e3)}
    if max_memory[-1] in units:
        unit = max_memory[-1].upper()
        max_memory_bytes = int(max_memory[:-1]) * units[unit]
    else:
        try:
            max_memory_bytes = int(max_memory) * units["M"]
        except ValueError:
            raise ValueError(
                """Could not parse max memory. Should be an int, optionally followed 
                with a unit {G, M, K}."""
            )
    max_batch_size = max_memory_bytes // seq_length
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
        "seq", shape=(n_seqs, seq_length), dtype=alphabet.dtype, compressor=compressor
    )

    rng = np.random.default_rng()

    logger.info("Adding sequences to Zarr store.")

    pbar = tqdm(unit="sequences", total=n_seqs)
    for i in range(0, n_seqs, max_batch_size):
        # don't go past n_seqs
        batch_size = min(max_batch_size, n_seqs - i)
        seq[i : i + batch_size] = rng.choice(alphabet, (batch_size, seq_length))
        pbar.update(batch_size)

    zarr.consolidate_metadata(output)  # type: ignore


if __name__ == "__main__":
    typer.run(main)
