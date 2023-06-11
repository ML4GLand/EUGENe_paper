import argparse
import torch
import tqdm
import seqdata as sd
import seqpro as sp

# Parse args
parser = argparse.ArgumentParser(description='Dataload sequences.')
parser.add_argument('--zarr_file', type=str, help='input zarr file')
parser.add_argument('--load', action='store_true', help='load the whole dataset into memory')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_batches', type=int, default=100, help='number of batches to process')

args = parser.parse_args()

# Load the data
sdata = sd.open_zarr(args.zarr_file)

# Load into mem if specified
if args.load:
    sdata.load()

# Create a dataloader
dl = sd.get_torch_dataloader(
    sdata,
    sample_dims="seq",
    variables=["seqs"],
    batch_size=args.batch_size,
    shuffle=False,
    transforms={"seqs": lambda x: torch.tensor(sp.ohe(x, alphabet=sp.ALPHABETS.DNA), dtype=torch.float32)}
)

# Loop through the dataloader in batches
for i, batch in tqdm.tqdm(enumerate(dl), total=args.num_batches):
    if i == args.num_batches:
        break
