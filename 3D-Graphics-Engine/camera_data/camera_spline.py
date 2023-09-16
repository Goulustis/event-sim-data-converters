import numpy as np
import scipy
import json
from config_cnsts import ALIAS_SCALE

# cam_spline_path = "camera_data/camera_spline.npy"
# intrinsics_path = "camera_data/ori_intrinsics.json"
cam_spline_path = "../synth_datapipeline/synthetic_ev_scene/camera_spline.npy"
intrinsics_path = "../synth_datapipeline/synthetic_ev_scene/intrinsics.json"

# intrinsics_path = "camera_data/intrinsics.json"


def read_intrinsics(intrxs_path, mult_factor = ALIAS_SCALE):
    with open(intrxs_path, "r") as f:
        data = json.load(f)

    cx = data["principal_point_x"]*mult_factor
    cy = data["principal_point_y"]*mult_factor
    fx = fy = data["focal_length"]*mult_factor

    return fx, fy, cx, cy

class CameraSpline:
    def __init__(self, filename=cam_spline_path, mode="smooth"):
        splinerep = np.load(filename, allow_pickle=True).item()
        self.eyerep = splinerep["eye"]
        self.targetrep = splinerep["target"]
        self.uprep = splinerep["up"]
        self.intrinsics = read_intrinsics(intrinsics_path)
        self.mode = mode

        self.get_dir_fn_dic = {"smooth":self.get_dir_smooth,
                               "lerp": self.get_dir_lerp}

        self.get_dir_fn = self.get_dir_fn_dic[self.mode]
    
    def get_dir_smooth(self, times):
        eye = np.stack(scipy.interpolate.splev(times, self.eyerep), axis=-1)
        target = np.stack(scipy.interpolate.splev(times, self.targetrep),
                          axis=-1)
        up = np.stack(scipy.interpolate.splev(times, self.uprep), axis=-1)
        return eye, target, up

    def get_dir_lerp(self, times):
        steps = 8
        t0 = np.floor(times * steps) / steps
        t1 = np.floor(times * steps + 1.0) / steps
        a = ((times - t0) / (t1 - t0))[..., None]
        eye0 = np.stack(scipy.interpolate.splev(t0, self.eyerep), axis=-1)
        eye1 = np.stack(scipy.interpolate.splev(t1, self.eyerep), axis=-1)
        eye = (1 - a) * eye0 + a * eye1
        target0 = np.stack(scipy.interpolate.splev(t0, self.targetrep), axis=-1)
        target1 = np.stack(scipy.interpolate.splev(t1, self.targetrep), axis=-1)
        target = (1 - a) * target0 + a * target1
        up0 = np.stack(scipy.interpolate.splev(t0, self.uprep), axis=-1)
        up1 = np.stack(scipy.interpolate.splev(t1, self.uprep), axis=-1)
        up = (1 - a) * up0 + a * up1

        return eye, target, up


    def interpolate(self, times):
        """Interpolate the camera spline at the given times.
        
        Args:
            times: A numpy array of shape (N, ) containing the times in
                microseconds.
        
        Returns:
            positions: A numpy array of shape (N, 3) containing the camera
                positions.
            rotations: A numpy array of shape (N, 3, 3) containing the world to
                camera rotations.
        """
        times = np.array(times, dtype=np.float64) / 1e7
        eye, target, up = self.get_dir_fn(times)

        forward = target - eye
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right, axis=-1, keepdims=True)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up, axis=-1, keepdims=True)

        return right, up, forward, eye
    
    def get_fow(self):
        """
        calculate field of view given intrinsics
        """
        fx, fy, cx, cy = self.intrinsics
        return 2*np.arctan(cy/fy)