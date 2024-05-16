import glob
import json
import os.path as osp
from tqdm import tqdm
import numpy as np

import argparse

def to_hom(mtx):
    u = np.eye(4)
    u[:3,:4] = mtx

    return u


def fix_camera(cam_dir):
    cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))

    for cam_f in tqdm(cam_fs, desc=f"fixing {osp.basename(osp.dirname(cam_dir))}"):
        with open(cam_f, "r") as f:
            cam_data = json.load(f)
        mtx_ori = np.array(cam_data["orientation"])
        pos = np.array(cam_data["position"]).reshape(3, 1)

        a, b, c = [mtx_ori[:, i] for i in range(3)]
        new_mtx = np.stack([-a, b, c], axis = 1)
        new_orientation = new_mtx.tolist()

        w2c = to_hom(np.concatenate([new_mtx, -mtx_ori@pos], axis=1))
        c2w = np.linalg.inv(w2c)
        new_pos = c2w[:3,3]
        
        cam_data["orientation"] = new_orientation
        cam_data["position"] = new_pos.tolist()

        with open(cam_f, "w") as f:
            json.dump(cam_data, f, indent=2)




if __name__ == "__main__":
    """
    cameras are mirror transform, flip the up vector so that it is not a mirror transform
    """
    # scene = "data/cat_fancy"
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", help="scene to fix camera for")
    args = parser.parse_args()
    
    scene = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/cat_fancy"
    cam_dirs = glob.glob(osp.join(scene, "*", "camera"))
    # cam_dirs = glob.glob(osp.join(args.scene, "*", "camera"))
    for cam_dir in cam_dirs:
        fix_camera(cam_dir)

    # fix_camera("/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/cat_fancy/colcam_set/camera")
