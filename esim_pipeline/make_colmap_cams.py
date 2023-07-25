from camera_spline import CameraSpline
from utils import make_camera, read_intrinsics_json
import numpy as np
import os.path as osp
import os
import json
import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import itertools

def get_triggers(cam_dir):

    fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))
    ts = []
    for cam_f in fs:
        with open(cam_f, "r") as f:
            ts.append(json.load(f)["t"])
    
    return np.array(ts)


def read_cam_params(cam_dir):
    fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))
    rots = []
    trans = []
    for cam_f in fs:
        with open(cam_f, "r") as f:
            data = json.load(f)
            rots.append(np.array(data["orientation"]))
            trans.append(np.array(data["position"]))
    
    return np.stack(rots), np.stack(trans)

def save_all_cams(orientations, poses, trig_ts, cam_dir, intrx_mtx):
    os.makedirs(cam_dir, exist_ok=True)

    for img_id, (ori, pos, trig_t) in enumerate(zip(orientations, poses, trig_ts)):
        cam_path = osp.join(cam_dir, f'{str(img_id).zfill(6)}.json')
        camera = make_camera(ori, pos, intrx_mtx)
        cam_json = camera.to_json()
        cam_json['t'] = str(trig_t)
        with open(cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)

def create_cams_at_colmap_ts():
    colmap_cam_dir = "/home/hunter/projects/ecam-cam-datapipline/adapt_carpet_colmap/colcam_set/camera"
    extrinsics_f = "generated_data/adapt_carpet/traj.txt"
    intrinsics_f = "generated_data/adapt_carpet/intrinsics.json"
    colcam_triggers = sorted(get_triggers(colmap_cam_dir)/1e3)
    camspline = CameraSpline(extrinsics_f)
    intrx = read_intrinsics_json(intrinsics_f)

    targ_dir = "camera_colmap_t"
    os.makedirs(targ_dir, exist_ok=True)

    cam_poses, orientations = camspline.interpolate(colcam_triggers)
    

    save_all_cams(orientations, cam_poses, colcam_triggers, targ_dir, intrx)


def plot_3d_points(gt_points, pred_points):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth points in red and label them
    # for i, point in enumerate(gt_points):
    #     ax.scatter(point[0], point[1], point[2], c='red')
    #     ax.text(point[0], point[1], point[2], str(i+1), color='red')

    # # Plot predicted points in blue and label them
    # for i, point in enumerate(pred_points):
    #     ax.scatter(point[0], point[1], point[2], c='blue')
    #     ax.text(point[0], point[1], point[2], str(i+1), color='blue')
    # Plot ground truth points in red
    ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='red', label='Ground Truth')

    # Plot predicted points in blue
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], c='blue', label='Predicted')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show the plot
    plt.show()


def plot_3d(points):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    for i, point in enumerate(points):
        ax.scatter(point[0], point[1], point[2], c='blue')
        ax.text(point[0], point[1], point[2], str(i+1), color='blue', fontsize=10)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def affine_registration(P, Q):
    """
    find T,t such that Q = P@T.T + t
    """
    transposed = False
    if P.shape[0] < P.shape[1]:
        transposed = True
        P = P.T
        Q = Q.T
    (n, dim) = P.shape
    # Compute least squares
    p, res, rnk, s = scipy.linalg.lstsq(np.hstack((P, np.ones([n, 1]))), Q)
    # Get translation
    t = p[-1][None]
    # Get transform matrix
    T = p[:-1].T
    # Compute transformed pointcloud
    Pt = P@T.T + t
    if transposed: Pt = Pt.T
    return Pt, (T, t)


def norm(a,b):
    return np.sum(np.sqrt((a-b)**2))

def find_colmap_esim_rotation_translation():
    colmap_cam_dir = "/home/hunter/projects/ecam-cam-datapipline/adapt_carpet_colmap/colcam_set/camera"
    esim_cam_colmap_t_cam_dir = "camera_colmap_t"
    
    colmap_rot, colmap_trans = read_cam_params(colmap_cam_dir)
    esim_rot, esim_trans = read_cam_params(esim_cam_colmap_t_cam_dir)

    colmap_trans = colmap_trans/norm(colmap_trans[1], colmap_trans[0])
    esim_trans = esim_trans/norm(esim_trans[1], esim_trans[0])

    
    fit_points, (T,t) = affine_registration(colmap_trans, esim_trans)




if __name__ == "__main__":
    # create_cams_at_colmap_ts()
    find_colmap_esim_rotation_translation()