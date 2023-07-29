import h5py
import numpy as np
import glob
import os.path as osp
import json


# def gen_colcam_triggers(rgb_dir, max_t = int(10*1e6), mode = "mid"):
#     """
#     generate mean time location of rgb frame
#     assume maxtime = 10 sec
#     units in micro sec;; 1sec = 1e6 microsec

#     mode (str): one of [start, mid, end] for trigger of starting time, center time and end time of a frame
#     """
#     n_frames = len(glob.glob(osp.join(rgb_dir, "*.png")))

#     if mode == "mid":
#         dt = -1/(2*n_frames)
#     elif mode == "start":
#         dt = -1/(n_frames)
#     else:
#         dt = 0
    
#     syn_ts = np.array(list(range(1,n_frames + 1)))/n_frames + dt
#     return (syn_ts * max_t).astype(int)
carpet_min_t = 84
carpet_max_t = 1382440.499
# def gen_colcam_triggers(rgb_dir:str = None, max_t = int(10*1e6), min_t = 0, mode:str = "mid", n_frames:int = None):
# def gen_colcam_triggers(rgb_dir:str = None, max_t = int(carpet_max_t), min_t = carpet_min_t, mode:str = "mid", n_frames:int = None):  # this for ecam traj
def gen_colcam_triggers(rgb_dir:str = None, max_t = int(10*1e6), min_t = 0, mode:str = "mid", n_frames:int = None, scene_mode="robo"):
    """
    generate mean time location of rgb frame
    assume maxtime = 10 sec
    units in micro sec;; 1sec = 1e6 microsec

    mode (str): one of [start, mid, end] for trigger of starting time, center time and end time of a frame
    """
    if scene_mode == "robo":
        max_t = int(10*1e6)
        min_t = 0
    elif scene_mode == "carpet":
        max_t = carpet_max_t
        min_t = carpet_min_t
        
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
