import glob
import json
import os.path as osp
from tqdm import tqdm

import argparse


def fix_camera(cam_dir):
    cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))

    for cam_f in tqdm(cam_fs, desc=f"fixing {osp.basename(osp.dirname(cam_dir))}"):
        with open(cam_f, "r") as f:
            cam_data = json.load(f)
        orientation = cam_data["orientation"]
        new_orientation = [orientation[0],
                           [-e for e in orientation[1]],
                           orientation[2]]

        cam_data["orientation"] = new_orientation

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

    # cam_dirs = glob.glob(osp.join(scene, "*", "camera"))
    cam_dirs = glob.glob(osp.join(args.scene, "*", "camera"))
    for cam_dir in cam_dirs:
        fix_camera(cam_dir)