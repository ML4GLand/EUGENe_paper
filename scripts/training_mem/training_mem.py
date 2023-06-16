from pathlib import Path

import typer


def main(
    store: Path,
    batch_size: int = 128,
    max_batches: int = 100,
    load: bool = False,
):
    import gc
    import logging
    import sys
    from time import perf_counter

    import numpy as np
    import seqdata as sd
    import seqpro as sp
    import torch
    import tqdm

    logger = logging.getLogger("DATALOADER")
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Load the data
    logger.info(f"Opening sdata: {str(store)}")
    sdata = sd.open_zarr(store)

    logger.info(f"Chunks: {sdata.seq.data.chunksize}")

    # Load into mem if specified
    if load:
        logger.info("Loading sdata into memory.")
        sdata.load()

    # Create a dataloader
    logger.info("Get dataloader.")
    dl = sd.get_torch_dataloader(
        sdata,
        sample_dims="_sequence",
        variables="seq",
        batch_size=batch_size,
        shuffle=False,
        transforms={
            "seq": lambda x: torch.tensor(
                sp.ohe(x, alphabet=sp.alphabets.DNA), dtype=torch.float32
            )
        },
    )

    # Loop through the dataloader in batches
    logger.info(f"Dataloader has {len(dl)} batches.")
    num_batches = min(max_batches, len(dl))

    logger.info(f"Iterating through {num_batches} batches.")
    timings = np.zeros(num_batches + 1, float)
    timings[0] = perf_counter()
    for i, batch in tqdm.tqdm(enumerate(dl), total=num_batches):
        del batch
        gc.collect()
        if i == max_batches:
            break
        timings[i + 1] = perf_counter()
    timings = np.diff(timings)

    if not load:
        output = store.parent / f"{store.stem}_timings.npy"
    else:
        output = store.parent / f"{store.stem}_timings_load.npy"
    logger.info(f"Saving timings to {output}")
    np.save(output, timings)


if __name__ == "__main__":
    typer.run(main)