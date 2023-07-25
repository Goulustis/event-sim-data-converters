from tqdm import tqdm
import h5py
import os.path as osp
import shutil
import numpy as np
import glob
import os

def create_clear_coarse(gap_size=45):
    st_idx = gap_size//2
    print("creating clear coarse")
    scene_path = "hosp_carpet"
    TARG_DIR = osp.join(scene_path, "clear_coarse_frames")
    # SRC_DIRS = [osp.join(scene_path, "fine_frames/gamma"), osp.join(scene_path, "fine_frames/linear")]
    SRC_DIRS = [osp.join(scene_path, "fine_frames/linear")]

    def cp_files(fs, targ_dir):
        os.makedirs(targ_dir, exist_ok=True)
        for i, f in enumerate(fs):
            dst_f = osp.join(targ_dir, f'{str(i).zfill(4)}.png')
            shutil.copy(f, dst_f)

    def write_trig(trigs, targ_dir):
        with open(osp.join(targ_dir, "triggers.txt"), "w") as f:
            for t in trigs:
                f.write(str(t) + "\n")

    for src_dir in SRC_DIRS:
        ver = osp.basename(src_dir)
        img_fs = np.array(sorted(glob.glob(osp.join(src_dir, "*.png"))))
        mv_idxs = np.array(list(range(len(img_fs))))[st_idx:][::gap_size]
        mv_img_fs = img_fs[mv_idxs]
        targ_dir = osp.join(TARG_DIR, ver)
        cp_files(mv_img_fs, targ_dir)

        triggers = gen_colcam_triggers(src_dir)[mv_idxs]
        write_trig(triggers, targ_dir)
    
    print("done clear coarse")


def gen_colcam_triggers(rgb_dir:str = None, max_t = int(1382440.499), min_t = 84, mode:str = "mid", n_frames:int = None):
    """ 
    generate mean time location of rgb frame
    assume maxtime = 10 sec
    units in micro sec;; 1sec = 1e6 microsec

    mode (str): one of [start, mid, end] for trigger of starting time, center time and end time of a frame
    """
    n_frames = len(glob.glob(osp.join(rgb_dir, "*.png"))) if n_frames is None else n_frames

    if mode == "mid":
        dt = -1/(2*n_frames)
    elif mode == "start":
        dt = -1/(n_frames)
    elif mode == "end":
        dt = 0 
    else:
        assert 0, f"{mode} not available"

    syn_ts = (np.array(list(range(1,n_frames + 1)))/n_frames + dt)*max_t
    syn_ts[0] = min_t if min_t < syn_ts[0] else syn_ts[0]
    return syn_ts


def comb_event_npys(event_dir):
    event_fs = sorted(glob.glob(osp.join(event_dir, "*.npy")))

    events = []

    for f in tqdm(event_fs, desc="loading events"):
        events.append(np.load(f))
    
    events = np.concatenate(events)
    xs, ys, ts, ps = [events[:,i] for i in range(4)]
    ts = (ts/1000).astype(np.uint64)
    
    targ_event_f = osp.join(event_dir, "events.hdf5")
    
    print(f"writing to: {targ_event_f}")
    with h5py.File(targ_event_f, "w") as hf:
        hf.create_dataset('x', data=xs, shape=xs.shape, dtype=np.uint16)
        hf.create_dataset('y', data=ys, shape=ys.shape, dtype=np.uint16)
        hf.create_dataset('t', data=ts, shape=ts.shape, dtype=np.uint64)
        hf.create_dataset('p', data=ps, shape=ps.shape, dtype=np.int8)


if __name__ == "__main__":
    # create_clear_coarse()
    comb_event_npys("/scratch/matthew/projects/enerf/data/enerf_adapt_carpet/events")