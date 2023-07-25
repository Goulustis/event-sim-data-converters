from camera_spline import CameraSpline
from utils import make_camera, read_intrinsics
import numpy as np
import os.path as osp
import os
import json
import glob

def get_triggers(cam_dir):

    fs = glob.glob(osp.join(cam_dir, "*.json"))
    ts = []
    for cam_f in fs:
        with open(cam_f, "r") as f:
            ts.append(json.load(f)["t"])
    
    return np.array(ts)


def save_all_cams(orientations, poses, trig_ts, cam_dir, intrx_mtx):
    os.makedirs(cam_dir, exist_ok=True)

    for img_id, (ori, pos, trig_t) in enumerate(zip(orientations, poses, trig_ts)):
        cam_path = osp.join(cam_dir, f'{img_id}.json')
        camera = make_camera(ori, pos, intrx_mtx)
        cam_json = camera.to_json()
        cam_json['t'] = str(trig_t)
        with open(cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)

def main():
    colcam_cam_dir = "/home/hunter/projects/ecam-cam-datapipline/adapt_carpet_colmap/colcam_set/camera"
    extrinsics_f = "generated_data/adapt_carpet/traj.txt"
    intrinsics_f = "generated_data/adapt_carpet/traj.txt"
    colcam_triggers = get_triggers(colcam_cam_dir)/1e3
    camspline = CameraSpline(extrinsics_f)
    intrx = read_intrinsics(intrinsics_f)

    targ_dir = "camera_colmap_t"
    os.makedirs(targ_dir, exist_ok=True)

    cam_poses, orientations = camspline.interpolate(colcam_triggers)
    
    cam_dir = osp.join(targ_dir, "camera")
    os.makedirs(cam_dir, exist_ok=True)

    save_all_cams(orientations, cam_poses, colcam_triggers, cam_dir, intrx)


if __name__ == "__main__":
    main()





