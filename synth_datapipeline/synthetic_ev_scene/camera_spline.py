import os.path as osp
import json
import glob
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d


# class CameraSpline:

#     def __init__(self, filename="synthetic_ev_scene/camera_spline.npy", mode = "smooth"):
#         splinerep = np.load(filename, allow_pickle=True).item()
#         self.eyerep = splinerep["eye"]
#         self.targetrep = splinerep["target"]
#         self.uprep = splinerep["up"]
#         self.mode = mode

#         self.get_dir_fn_dic = {"smooth":self.get_dir_smooth,
#                                "enerf": self.get_dir_smooth,
#                                "lerp": self.get_dir_lerp}

#         self.get_dir_fn = self.get_dir_fn_dic[self.mode]
    
#     def get_dir_smooth(self, times):
#         eye = np.stack(scipy.interpolate.splev(times, self.eyerep), axis=-1)
#         target = np.stack(scipy.interpolate.splev(times, self.targetrep),
#                           axis=-1)
#         up = np.stack(scipy.interpolate.splev(times, self.uprep), axis=-1)
#         return eye, target, up

#     def get_dir_lerp(self, times):
#         steps = 48
#         t0 = np.floor(times * steps) / steps
#         t1 = np.floor(times * steps + 1.0) / steps
#         a = ((times - t0) / (t1 - t0))[..., None]
#         eye0 = np.stack(scipy.interpolate.splev(t0, self.eyerep), axis=-1)
#         eye1 = np.stack(scipy.interpolate.splev(t1, self.eyerep), axis=-1)
#         eye = (1 - a) * eye0 + a * eye1
#         target0 = np.stack(scipy.interpolate.splev(t0, self.targetrep), axis=-1)
#         target1 = np.stack(scipy.interpolate.splev(t1, self.targetrep), axis=-1)
#         target = (1 - a) * target0 + a * target1
#         up0 = np.stack(scipy.interpolate.splev(t0, self.uprep), axis=-1)
#         up1 = np.stack(scipy.interpolate.splev(t1, self.uprep), axis=-1)
#         up = (1 - a) * up0 + a * up1

#         return eye, target, up


#     def interpolate(self, times, model="hyper"):
#         """Interpolate the camera spline at the given times.
        
#         Args:
#             times: A numpy array of shape (N, ) containing the times in
#                 microseconds.
        
#         Returns:
#             positions: A numpy array of shape (N, 3) containing the camera
#                 positions.
#             rotations: A numpy array of shape (N, 3, 3) containing the world to
#                 camera rotations.
#         """
#         times = np.array(times, dtype=np.float64) / 1e7
#         # eye = np.stack(scipy.interpolate.splev(times, self.eyerep), axis=-1)
#         # target = np.stack(scipy.interpolate.splev(times, self.targetrep),
#         #                   axis=-1)
#         # up = np.stack(scipy.interpolate.splev(times, self.uprep), axis=-1)
#         eye, target, up = self.get_dir_fn(times)

#         forward = target - eye
#         forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
#         right = np.cross(forward, up)
#         right /= np.linalg.norm(right, axis=-1, keepdims=True)
#         up = np.cross(right, forward)
#         up /= np.linalg.norm(up, axis=-1, keepdims=True)


#         if model == "enerf":
            
#             rotation = np.stack([-right, up, forward], axis=-1)
#         else:
#             rotation = np.stack([-right, -up, forward], axis=-2)

#         return eye, rotation

class CameraSpline:

    def __init__(self, filename="synthetic_ev_scene/camera_spline.npy", mode = "smooth"):
        splinerep = np.load(filename, allow_pickle=True).item()
        self.eyerep = splinerep["eye"]
        self.targetrep = splinerep["target"]
        self.uprep = splinerep["up"]
        self.mode = mode

        self.get_dir_fn_dic = {"smooth":self.get_dir_smooth,
                               "enerf": self.get_dir_smooth,
                               "lerp": self.get_dir_lerp}

        self.get_dir_fn = self.get_dir_fn_dic[self.mode]
    
    def get_dir_smooth(self, times):
        eye = np.stack(scipy.interpolate.splev(times, self.eyerep), axis=-1)
        target = np.stack(scipy.interpolate.splev(times, self.targetrep),
                          axis=-1)
        up = np.stack(scipy.interpolate.splev(times, self.uprep), axis=-1)
        return eye, target, up

    def get_dir_lerp(self, times):
        steps = 48
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


    def interpolate(self, times, model="hyper"):
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
        # eye = np.stack(scipy.interpolate.splev(times, self.eyerep), axis=-1)
        # target = np.stack(scipy.interpolate.splev(times, self.targetrep),
        #                   axis=-1)
        # up = np.stack(scipy.interpolate.splev(times, self.uprep), axis=-1)
        eye, target, up = self.get_dir_fn(times)

        forward = target - eye
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right, axis=-1, keepdims=True)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up, axis=-1, keepdims=True)


        if model == "enerf":
            
            rotation = np.stack([-right, up, forward], axis=-1)
        else:
            rotation = np.stack([-right, -up, forward], axis=-2)

        return eye, rotation

class CameraSpline_enerf:

    def __init__(self, src_dir):
        cam_dir = osp.join(src_dir, "ecam_set", "camera")
        colcam_dir = osp.join(src_dir, "colcam_set", "camera")
        rots, locs, ts = self._read_cameras(cam_dir)
        col_rots, col_locs, col_ts = self._read_cameras(colcam_dir)
        rots, locs, ts = np.concatenate([rots, col_rots]), np.concatenate([locs, col_locs]), np.concatenate([ts, col_ts])
        idxs = np.argsort(ts)
        rots, locs, ts = [e[idxs] for e in [rots, locs, ts]]

        cond = np.concatenate([(np.diff(ts) > 0), np.array([not (ts[-1] == ts[-2])])])

        self.ts = ts[cond]
        self.locs = locs[cond]
        self.rots = rots[cond]

        # right_neg, up_neg, forward = [self.rots[:,i, :] for i in range(3)] # hypernerf is [-right, -up, forward]
        # self.rots = np.stack([right_neg, -up_neg, forward], axis = -1) # enerf is [-right, up, forward]

        ## IMPORTANT: COMMENT OUT
        self.locs[:,2] = self.locs[:,2]+0.9
        
        # self.rot_interpolator = Slerp(self.ts, Rotation.from_matrix(self.rots))
        # self.trans_interpolator = interp1d(self.ts, self.locs, axis=0, kind="cubic")

        # assert (np.abs((self.rot_interpolator(self.ts).as_matrix() - self.rots).reshape(-1)) < 1e-4).all(), "mirror transform!"
        self._setup_interpolators()
    

    def _setup_interpolators(self):
        self.trans_interpolator = interp1d(self.ts, self.locs, axis=0, kind="cubic")

        right_neg, up_neg, forward = [self.rots[:,i, :] for i in range(3)] # hypernerf is [-right, -up, forward]
        for i in range(8):
            b_num = f'{i:08b}'
            a, b, c = [int(e) for e in b_num[-3:]]
            self.rots = np.stack([right_neg*(-1)**(a), up_neg*(-1)**(b), forward], axis = -1)
            self.rot_interpolator = Slerp(self.ts, Rotation.from_matrix(self.rots))

            if (np.abs((self.rot_interpolator(self.ts).as_matrix() - self.rots).reshape(-1)) < 1e-4).all():
                break
        
        assert (np.abs((self.rot_interpolator(self.ts).as_matrix() - self.rots).reshape(-1)) < 1e-4).all(), "mirror transform!"

    def _read_cameras(self, cam_dir):
        cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))

        rots, locs, ts = [], [], []
        for f in cam_fs:
            with open(f, "r") as f:
                data = json.load(f)
                rots.append(np.array(data["orientation"]))
                locs.append(np.array(data["position"]))
                ts.append(float(data["t"]))

        rots = np.stack(rots)
        # right_neg, up_neg, forward = [rots[:,i, :] for i in range(3)] # hypernerf is [-right, -up, forward]
        # rots = np.stack([right_neg, -up_neg, forward], axis = -1) # enerf is [-right, up, forward]
        return np.stack(rots), np.stack(locs), np.array(ts)
        
        

    def interpolate(self, times, model="enerf"):
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
        return self.trans_interpolator(times), self.rot_interpolator(times).as_matrix()


    def triggers(self, n_samples=None):
        # if n_samples is None:
        #     n_samples = len(self.ts)
        # min_t, max_t = self.ts.min(), self.ts.max()
        # gap = (max_t - min_t)/n_samples
        # return np.arange(start=min_t, stop=max_t, step=gap)
        return self.ts
        # return np.concatenate([np.arange(start=min_t, stop=max_t, step=gap), np.array([max_t])])

if __name__ == "__main__":
    cs = CameraSpline("synthetic_scene/camera_spline.npy")
    eye, rot = cs.interpolate(np.linspace(0, 1e7, 10))
    print(np.einsum("nij,nj->ni", rot, eye))
