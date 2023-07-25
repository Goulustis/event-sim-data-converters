import numpy as np
from camera_spline import CameraSpline
import glob
import os.path as osp
import os
import json
from utils import read_intrinsics
from format_ecam_set import create_and_write_camera_extrinsics

def read_time(json_f):
    with open(json_f, "r") as f:
        data = json.load(f)
    
    return data["t"]

def get_ts(cam_dir, n_split = 30):
    json_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))

    idx = 10#len(cam_dir) // 2
    st_json_f = json_fs[idx]
    end_json_f = json_fs[idx + 1]

    st_t, end_t = read_time(st_json_f), read_time(end_json_f)

    return np.linspace(st_t, end_t, n_split)

    
    


def main():
    camera_dir = "ShakeCarpet1_formatted/ecam_set/camera"
    extrinsics_f = "ori_data/ShakeCarpet1/poses_all.txt"
    intrinsics_f = "ori_data/ShakeCarpet1/poses_bounds.npy"

    intrxs = read_intrinsics(intrinsics_f)
    spline = CameraSpline(extrinsics_f)
    ts = get_ts(camera_dir)
    poses, orientations = spline.interpolate(ts)

    create_and_write_camera_extrinsics("/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/ShakeCarpet1_formatted/colcam_set/camera-paths/fine-ecam-traj", 
                                       orientations, poses, ts, intrxs)


if __name__ == "__main__":
    main()
