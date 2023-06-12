from pathlib import Path

import typer


def main(
    store: Path,
    batch_size=128,
    num_batches=100,
    load=False,
):
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
    logger.info('Opening sdata.')
    sdata = sd.open_zarr(store)
    
    logger.info(f'Chunks: {sdata.seq.chunksizes}')

    # Load into mem if specified
    if load:
        logger.info('Loading sdata into memory.')
        sdata.load()

    # Create a dataloader
    logger.info('Get dataloader.')
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
    logger.info(f'Iterating through {num_batches} batches.')
    timings = np.zeros(num_batches+1, float)
    timings[0] = perf_counter()
    for i, batch in tqdm.tqdm(enumerate(dl), total=num_batches):
        timings[i+1] = perf_counter()
        if i == num_batches:
            break
    timings = np.diff(timings)
    
    if not load:
        output = f'{store.stem}_timings.npy'
    else:
        output = f'{store.stem}_timings_load.npy'
    logger.info(f'Saving timings to {output}')
    np.save(output, timings)


if __name__ == "__main__":
    typer.run(main)
