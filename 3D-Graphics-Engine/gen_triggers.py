import numpy as np
import glob
import os.path as osp
import os
from engine_configs import SCENE

def gen_colcam_triggers(rgb_dir:str = None, max_t = int(10*1e6), min_t = 0, mode:str = "mid", n_frames:int = 4096):
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
    syn_ts[0] = min_t
    return syn_ts


def create_txt_triggers(n_frames, dst_path = "triggers.txt", max_t:int = int(10*1e6), min_t = 0):

    trig_ts = gen_colcam_triggers(max_t=max_t, mode="start", n_frames=n_frames, min_t = min_t)

    with open(dst_path, "w") as f:
        for t in trig_ts:
            f.write(str(int(t)) + "\n")

    print("done creating triggers")


def generate_triggers(n_frames=2048, max_t:int = int(10*1e6), min_t = 0):
    dst_path = "camera_data/triggers.txt"
    if osp.exists(dst_path):
        os.remove(dst_path)

    if SCENE == "robo":
        create_txt_triggers(n_frames, dst_path)
    elif SCENE == "carpet":
        create_txt_triggers(n_frames, dst_path, max_t = max_t, min_t = min_t)
    else:
        raise Exception(f"{SCENE} not supported")



if __name__ == "__main__":
    n_frames = 2048

    if SCENE == "robo":
        generate_triggers(n_frames)
    elif SCENE == "carpet":
        carpet_min_t = 84
        carpet_max_t = 1382440.499
        generate_triggers(n_frames, max_t=carpet_max_t, min_t = carpet_min_t) # generate carpet ts
    else:
        raise Exception(f"{SCENE} not supported")
    
    print(f"created {n_frames} {SCENE} triggers")