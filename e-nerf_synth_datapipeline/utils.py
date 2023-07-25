import h5py
import numpy as np
import glob
import os.path as osp
import json
from camera_spline import CameraSpline
from nerfies.camera import Camera

class EventBuffer:
    def __init__(self, ev_f) -> None:
        self.f = h5py.File(ev_f, "r")
        self.x_f = self.f["x"]
        self.y_f = self.f["y"]
        self.p_f = self.f["p"]
        self.t_f = self.f["t"]

        self.fs = [self.x_f, self.y_f, self.p_f, self.t_f]

        self.n_retrieve = 100000
        self.x_cache = np.array([self.x_f[0]])
        self.y_cache = np.array([self.y_f[0]])
        self.t_cache = np.array([self.t_f[0]])
        self.p_cache = np.array([self.p_f[0]])

        self.caches = [self.x_cache, self.y_cache, self.t_cache, self.p_cache]

        self.curr_pnter = 1
    
    def update_cache(self):
        
        rx, ry, rp, rt = [e[self.curr_pnter:self.curr_pnter + self.n_retrieve] for e in self.fs]
        self.x_cache = np.concatenate([self.x_cache, rx])
        self.y_cache = np.concatenate([self.y_cache, ry])
        self.p_cache = np.concatenate([self.p_cache, rp])
        self.t_cache = np.concatenate([self.t_cache, rt])
        
        self.curr_pnter = min(len(self.t_f), self.curr_pnter + self.n_retrieve)

    def drop_cache_by_cond(self, cond):
        self.x_cache = self.x_cache[cond]
        self.y_cache = self.y_cache[cond]
        self.p_cache = self.p_cache[cond]
        self.t_cache = self.t_cache[cond]

    def retrieve_data(self, st_t, end_t):
        while self.t_cache[-1] <= end_t and (self.curr_pnter < len(self.t_f)):
            self.update_cache()
        
        ret_cond = ( st_t<= self.t_cache) & (self.t_cache <= end_t)
        ret_data = [self.t_cache[ret_cond], self.x_cache[ret_cond], self.y_cache[ret_cond], self.p_cache[ret_cond]]
        self.drop_cache_by_cond(~ret_cond)

        return ret_data

##################################
# Reading
##################################
def read_poses_bounds(path_poses_bounds, start_frame=None, end_frame=None, skip_frames=None, invert=False):
    """ Returns: 
    #    :poses np.array (num_poses, 3, 5) in  c2w (even for esim, these poses are already inverted to c2w)
         :bds (num_poses, 2) where [:, 0] = min_depth, and [:, 1] = max_depth
    """
    assert osp.exists(path_poses_bounds)

    poses_arr = np.load(path_poses_bounds)
    assert poses_arr.shape[0] > 10
    assert poses_arr.shape[1] == 17
    # (num_poses, 17), where  17 = (rot | trans | hwf).ravel(), zmin, zmax
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (num_poses, 17) to (num_poses, 3, 5)
    bds = poses_arr[:, -2:]  # (num_poses, 2)
    num_poses = poses.shape[0]

    if invert:
        for i in range(num_poses):
            rot, trans = poses[i, :3, :3], poses[i, :, 3]
            rot, trans = invert_trafo(rot, trans)
            poses[i, :3, :3] = rot
            poses[i, :3, 3] = trans
        print("** Inverted Poses from ** ", path_poses_bounds)

    # check rotation matrix
    for i in range(num_poses):
        rot = poses[i, :3, :3]
        check_rot(rot, right_handed=True)

    if (start_frame is not None) and (end_frame is not None) and (skip_frames is not None):
        assert end_frame > start_frame
        assert start_frame >= 0
        assert skip_frames > 0
        assert skip_frames < 50

        if end_frame == -1:
            end_frame = poses.shape[0] - 1

        poses = poses[start_frame:end_frame:skip_frames, ...]  # (num_poses, 3, 5)
        bds = bds[start_frame:end_frame:skip_frames, :]

    print("Got total of %d poses" % (poses.shape[0]))
    return poses, bds


def check_rot(rot, right_handed=True, eps=1e-6):
    """
    Input: 3x3 rotation matrix
    """
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3

    assert np.allclose(rot.transpose() @ rot, np.eye(3), atol=1e-6)
    assert np.linalg.det(rot) - 1 < eps * 2

    if right_handed:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) - 1.0) < 1e-3
    else:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) + 1.0) < 1e-3

def invert_trafo(rot, trans):
    # invert transform from w2cam (esim, colmap) to cam2w
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3
    assert trans.shape[0] == 3

    rot_ = rot.transpose()
    trans_ = -1.0 * np.matmul(rot_, trans)

    check_rot(rot_)
    return rot_, trans_


def make_camera(R, t, intr_mtx):
    """
    input:
        ext_mtx (np.array): World to cam matrix - shape = 4x4
        intr_mtx (np.array): intrinsic matrix of camera - shape = 3x3

    return:
        nerfies.camera.Camera of the given mtx
    """
   
    k1, k2, p1, p2, k3 = [0,0,0,0,0]
    coord = t

    cx, cy = intr_mtx[:2,2].astype(int)

    new_camera = Camera(
        orientation=R,
        position=coord,
        focal_length=intr_mtx[0,0],
        pixel_aspect_ratio=1,
        principal_point=np.array([cx, cy]),
        radial_distortion=(k1, k2, 0),
        tangential_distortion=(p1, p2),
        skew=0,
        image_size=np.array([2*cx, 2*cy])  ## (width, height) of camera
    )

    return new_camera

def extract_image_ids(img_dir):
    img_fs = sorted(glob.glob(osp.join(img_dir, "*.png")))
    return [osp.basename(f).split(".")[0] for f in img_fs]

def read_point_clouds(file_path):

    points = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            u = line.split(" ")[:3]
            u = [float(e) for e in u]
            points.append(np.array(u))
    
    return np.stack(points)

def read_evs_h5(file_path="synthetic_ev_scene/events.hdf5"):
    print("loading hdf5")
    with h5py.File(file_path, "r") as f:
        xs,ys,ts,ps = [f[e][:] for e in list("xytp")]
        
    return {"x": xs, "y":ys, "t":ts, "p":ps}


def read_evs_h5_fast(file_path="synthetic_ev_scene/events.hdf5"):
    print("loading hdf5")
    with h5py.File(file_path, "r") as f:
        xs,ys,ts,ps = [f[e] for e in list("xytp")]
        
    return {"x": xs, "y":ys, "t":ts, "p":ps}

def gen_colcam_triggers(rgb_dir, max_t = int(10*1e6), mode = "mid"):
    """
    generate mean time location of rgb frame
    assume maxtime = 10 sec
    units in micro sec;; 1sec = 1e6 microsec

    mode (str): one of [start, mid, end] for trigger of starting time, center time and end time of a frame
    """
    n_frames = len(glob.glob(osp.join(rgb_dir, "*.png")))

    if mode == "mid":
        dt = -1/(2*n_frames)
    elif mode == "start":
        dt = -1/(n_frames)
    else:
        dt = 0
    
    syn_ts = (np.array(list(range(1,n_frames + 1)))/n_frames + dt)*max_t
    return syn_ts 

def read_triggers(trig_f):
    triggers = []
    with open(trig_f, "r") as f:
        for line in f:
            line = line.strip()
            triggers.append(round(float(line)))
        
    return np.array(triggers).astype(np.uint32)


def read_intrinsics(f):
    poses_c2w, bds = read_poses_bounds(f)
    hwf = poses_c2w[0, :3, -1]

    H, W, focal = hwf
    H, W = int(H), int(W)
    cx = W / 2.0
    cy = H / 2.0
    return np.array([[focal, 0, cx],
                     [0, focal, cy],
                     [0, 0,      1]])


def read_intrinsics_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    cx = data["principal_point_x"]
    cy = data["principal_point_y"]
    fx = fy = data["focal_length"]

    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


    

def gen_cams(eimg_ts:np.ndarray, cam_generator:CameraSpline):
    return cam_generator.interpolate(eimg_ts)

def read_triggers(trig_f):
    triggers = []
    with open(trig_f, "r") as f:
        for line in f:
            line = line.strip()
            triggers.append(round(float(line)))

    return np.array(triggers).astype(np.uint32)

def read_json(json_f):
    with open(json_f, "r") as f:
        return json.load(f)