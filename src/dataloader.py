import os
import glob
import numpy as np
import torch

script_dir = os.path.dirname(__file__)

def _candidate_roots():
    """Return a prioritized list of dataset dirs to search."""
    cands = []
    # explicit env overrides win
    env_dir = os.environ.get("DATA_DIR") or os.environ.get("EDU_FW_DIR")
    if env_dir:
        cands.append(os.path.abspath(env_dir))
    # common repo-relative defaults
    cands += [
        os.path.abspath(os.path.join(script_dir, "../data/edu_fineweb10B")),
        os.path.abspath(os.path.join(script_dir, "../data/fineweb10B")),
        os.path.abspath(os.path.join(script_dir, "../../dev/data/edu_fineweb10B")),
        os.path.abspath(os.path.join(script_dir, "../../dev/data/fineweb10B")),
    ]
    # keep only existing dirs, preserve order
    seen = set()
    out = []
    for p in cands:
        if os.path.isdir(p) and p not in seen:
            out.append(p); seen.add(p)
    return out

def _list_shards(root, split):
    """Return sorted list of shard paths matching split in name, across extensions."""
    patterns = [f"*{split}*.npy", f"*{split}*.npz", f"*{split}*.bin"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat)))
    # stable, human-friendly order
    files = sorted(files)
    return files

class DataLoaderLite:
    """A simple dataloader for pre-tokenized FineWeb shards (.npy or uint16 .bin)."""

    def __init__(self, B, T, process_rank, num_processes, split='train'):
        super().__init__()
        assert split in {'train', 'val'}
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # discover shards
        roots = _candidate_roots()
        shard_paths = []
        self.data_root = None
        for r in roots:
            fps = _list_shards(r, split)
            if fps:
                self.data_root = r
                shard_paths = fps
                break

        if not shard_paths:
            msg = [
                f"No shards found for split '{split}'.",
                "Searched:",
                *[f"  - {r}" for r in roots] if roots else ["  - <no existing candidate dirs>"],
                "Expect filenames containing 'train' or 'val' and extension .npy/.npz/.bin",
                "Tip: set DATA_DIR=/scratch/group/csce689609/data/edu_fineweb10B (or fineweb10B).",
                "      or create a symlink at ./data/edu_fineweb10B → shared folder.",
            ]
            raise FileNotFoundError("\n".join(msg))

        master = (process_rank == 0)
        if master:
            print(f"Using dataset root: {self.data_root}")
            print(f"Found {len(shard_paths)} shards for split '{split}'. Example: {os.path.basename(shard_paths[0])}")

        self.shard_filepaths = shard_paths
        self.reset()

    def load_tokens(self, filepath: str) -> torch.Tensor:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".npy",):
            arr = np.load(filepath, mmap_mode="r")  # uint16 or int32 expected
            if hasattr(arr, "dtype") and arr.dtype != np.int32:
                arr = arr.astype(np.int32, copy=False)
        elif ext in (".npz",):
            npz = np.load(filepath, mmap_mode="r")
            key = npz.files[0]  # assume single array
            arr = npz[key]
            if arr.dtype != np.int32:
                arr = arr.astype(np.int32, copy=False)
        elif ext == ".bin":
            # raw contiguous uint16 token stream (as used by many FW dumps)
            arr = np.fromfile(filepath, dtype=np.uint16).astype(np.int32, copy=False)
        else:
            raise ValueError(f"Unsupported shard type: {filepath}")
        return torch.as_tensor(arr, dtype=torch.long)

    def reset(self):
        self.curr_shard = 0
        self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
        # rank-strided start (DDP sharding in-place)
        self.curr_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        need = B * T + 1
        # if not enough tokens left in this shard for all ranks, roll to next
        if self.curr_pos + need > len(self.tokens):
            self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
            self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
            self.curr_pos = self.B * self.T * self.process_rank

        batch = self.tokens[self.curr_pos : self.curr_pos + need]
        x_batch = batch[:-1].view(B, T)
        y_batch = batch[1:].view(B, T)
        # advance by all-process stride
        self.curr_pos += B * T * self.num_processes
        return x_batch, y_batch
