import numpy as np
import json
import jax
from tqdm import tqdm

def tree_collate(list_of_pytrees):
  """Collates a list of pytrees with the same structure."""
  return jax.tree_map(lambda *x: np.stack(x), *list_of_pytrees)

EIMG_SIZE = (286, 360, 480)
N, H, W = EIMG_SIZE
IMG_AREA = H*W
IDXS=[i*IMG_AREA + IMG_AREA//2 + W//2 for i in range(N)]

def load_metadata():
    metadata_f = "ShakeCarpet1_formatted/ecam_set/metadata.json"
    with open(metadata_f, "r") as f:
        data = json.load(f)
    
    data = jax.tree_map(lambda x : np.atleast_1d(x), data)
    data_ls = []
    for k, v in data.items():
        data_ls.append(v)
    
    data_ls = tree_collate(data_ls)
    
    return data_ls

def read_ts():
    metadata = load_metadata()
    n_pix = IMG_AREA

    ts = []
    for idx in IDXS:
        d1_idx = idx//n_pix
        t = metadata['t'][d1_idx]
        ts.append(t)
    
    return ts

def main():
    cy, cx = [240, 180]
    val_ts = read_ts()
    # (x, y, p, t, t_prev, next_idx)
    lnk_evs = np.load("ShakeCarpet1_formatted/ecam_set/linked_events.npy")
    xs, ys, ts, ps = [lnk_evs[k] for k in list("xytp")]

    val_idxs = []
    val_t_samples = []
    for i, t in tqdm(enumerate(val_ts), total=len(val_ts)):
        cond = (ts >= t) & (ts <= t+5000) & (xs == cx) & (ys == cy)
        pval = ps[cond].sum()

        val_idx = cond.argmax() if pval != 0 else -1
        ev = lnk_evs[val_idx]
        t_sample = 5000 + (t - ev["t_prev"]).item() if pval != 0 else -1

        val_idxs.append(val_idx)
        val_t_samples.append(t_sample)

    np.save("val_idxs.npy", np.array(val_idxs))
    np.save("val_t_samples.npy", np.array(val_t_samples))


if __name__ == "__main__":
    main()
    

    