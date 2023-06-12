from pathlib import Path

import typer


def main(
    store: Path,
):
    import logging
    import sys
    
    import numcodecs as ncds
    import zarr
    from rechunker import rechunk
    
    logger = logging.getLogger("GENERATE_SDATA")
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    mem_per_chunk = int(1028**3)
    root = zarr.open_group(store)
    logging.info(f'Sequences have shape: {root["seq"].shape}')
    
    length = root['seq'].shape[1]
    seqs_per_chunk = min(mem_per_chunk // length, 4096)
    logging.info(f'Rechunking to have {seqs_per_chunk} sequences per chunk')
    
    plan = rechunk(
        zarr.open(store),
        target_chunks={'seq': {'_sequence': seqs_per_chunk, '_length': length}},
        max_mem="40GB",
        target_store=f'{store.parent / store.stem}_rechunked.zarr',
        target_options={'seq': {'compressor': ncds.Blosc('zstd', clevel=7, shuffle=-1)}}
    )
    plan.execute()
    
if __name__ == '__main__':
    typer.run(main)