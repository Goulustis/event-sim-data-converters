import sys
sys.path.append(".")

import numpy as np
from tqdm import tqdm
from utils import read_evs_h5, gen_colcam_triggers
import os.path as osp


linked_Type = np.dtype({'names':['x','y','p','t', "t_prev", "next_idx"], 
                        'formats':['<u2','<u2','i2', 'u4', 'u4', 'i4']})

def get_start_idx(ts, st_t):

    for idx, t in enumerate(ts):
        if st_t <= t:
            return idx

    return -1

def main():
    """
    turn all events into tuple of: (x, y, p, t, t_prev, next_idx),
                                   next_idx = -1 if event is last event
    """
    rgb_dir = "synthetic_ev_scene/coarse_frames/gamma"
    ev_path = "synthetic_ev_scene/events.hdf5"

    trigger_st = gen_colcam_triggers(rgb_dir)[0]
    evs = read_evs_h5(ev_path)
    xs, ys ,ts, ps = [evs[e] for e in list("xytp")]
    h, w = 1080, 1920

    keep_cond = ts > trigger_st
    xs, ys, ts, ps = [e[keep_cond] for e in [xs, ys, ts, ps]]
    st_idx = get_start_idx(ts, trigger_st)
    assert st_idx != -1, "time not found"

    prev_ts = np.full((h, w), trigger_st)
    prev_idxs = np.full((h, w), -1)
    # lined_data = []
    lined_data = np.empty(len(xs), dtype=linked_Type)

    curr_idx = 0
    for _, (x,y,t,p) in tqdm(enumerate(zip(xs,ys,ts,ps)), total=len(xs), desc="lining data"):
    # for curr_idx in tqdm(range(st_idx, len(ts))):
        # x, y, t, p = xs[curr_idx], ys[curr_idx], ts[curr_idx], ps[curr_idx]
        prev_t = prev_ts[y, x]
        curr_ev = np.array([(x, y, p, t, prev_t, -1)], dtype=linked_Type)

        # prev_idx = prev_idxs[y, x]
        # if prev_idx != -1:
        #     prev_ev = lined_data[prev_idx]
        #     prev_ev[-1] = curr_idx
        # prev_idxs[y, x] = curr_idx

        # lined_data[curr_idx] = curr_ev

        prev_idx = prev_idxs[y, x]
        if prev_idx != -1:
            prev_ev = lined_data[prev_idx]
            prev_t = prev_ev["t"]
            
            if prev_t != t:
                prev_idxs[y, x] = curr_idx
                prev_ts[y, x] = t
                prev_ev[-1] = curr_idx
                lined_data[curr_idx] = curr_ev
                curr_idx += 1
            else:
                pass
        else:
            prev_idxs[y, x] = curr_idx
            prev_ts[y, x] = t
            lined_data[curr_idx] = curr_ev
            curr_idx += 1
    
    lined_data = lined_data[:curr_idx]
    save_path = "linked_events.npy"
    np.save(save_path, lined_data)


if __name__ == "__main__":
    main()
