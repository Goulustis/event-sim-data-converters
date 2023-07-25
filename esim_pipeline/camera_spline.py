import numpy as np
np.set_printoptions(2)
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d



class CameraSpline:
    def __init__(self, extrxs_path):
        data = np.loadtxt(extrxs_path, skiprows=1, delimiter=" ")
        self.ts = data[:,0]/1e3
        self.pos = data[:,1:4].reshape(len(self.ts), 3)
        self.quats = data[:, 4:].reshape(len(self.ts), 4) # w2c

        cond = np.concatenate([(np.diff(self.ts) > 0), np.array([True])])
        self.ts = self.ts[cond]
        self.pos = self.pos[cond]
        self.quats = self.quats[cond]


        self.rot_interpolator = Slerp(self.ts, Rotation.from_quat(self.quats))
        self.trans_interpolator = interp1d(x=self.ts, y=self.pos, axis=0, kind="cubic", bounds_error=True)
    
    def interpolate(self, t):
        # cond = t < self.min_t
        # t[cond] = self.min_t
        rs = self.rot_interpolator(t).as_matrix()
        if len(rs.shape) == 3:
            rs = rs.transpose(0,2,1)
        else:
            rs = rs.T

        return self.trans_interpolator(t), rs

    def interp_mtx(self, t):
        trans, rot = self.interpolate(t)
        return np.concatenate([rot, trans], axis=-1)

