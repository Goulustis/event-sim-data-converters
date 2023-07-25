import numpy as np

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

def get_hom_trafos(rots_3_3, trans_3_1):
    N = rots_3_3.shape[0]
    assert rots_3_3.shape == (N, 3, 3)

    if trans_3_1.shape == (N, 3):
        trans_3_1 = np.expand_dims(trans_3_1, axis=-1)
    else:
        assert trans_3_1.shape == (N, 3, 1)
    
    pose_N_4_4 = np.zeros((N, 4, 4))
    hom = np.array([0,0,0,1]).reshape((1, 4)).repeat(N, axis=0).reshape((N, 1, 4))

    pose_N_4_4[:N, :3, :3] = rots_3_3  # (N, 3, 3)
    pose_N_4_4[:N, :3, 3:4] = trans_3_1 # (N, 3, 1)
    pose_N_4_4[:N, 3:4, :] = hom # (N, 1, 4)

    # pose_N_3_4 = np.asarray([np.concatenate((r, t), axis=1) for r, t in zip(rots_3_3, trans_3_1)])
    # pose_N_4_4 = np.asarray([np.vstack((p, np.asarray([0, 0, 0, 1]))) for p in pose_N_3_4])
    return pose_N_4_4

def quatList_to_poses_hom_and_tss(quat_list_us):
    """
    quat_list: [[t, px, py, pz, qx, qy, qz, qw], ...]

    return:
        tss_all_poses_us (int): time stamp in us
        all_trafos (np.ndarray): 4x4 extrinsics (C2W)
    """
    # thing is in nano seconds, need to divide the thing to get correct time scale
    tss_all_poses_us = quat_list_us[:,0]/1e3

    all_rots = [Rotation.from_quat(rot[4:]).as_matrix() for rot in quat_list_us]
    all_trans = [trans[1:4] for trans in quat_list_us]
    all_trafos = get_hom_trafos(np.asarray(all_rots), np.asarray(all_trans))

    return tss_all_poses_us, all_trafos


def blender_2_opencv(c2m_mtrxs):
    blender2cv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    
    cv_c2ws = np.stack([np.matmul(cam_world,blender2cv) for cam_world in c2m_mtrxs])
    world_coords = cv_c2ws[:,:3,3]
    w2cs = np.stack(np.linalg.inv(cv_c2w) for cv_c2w in cv_c2ws)
    
    return w2cs, world_coords

class CameraSpline:
    def __init__(self, extrxs_path):
        self.ts, self.extrnsics = quatList_to_poses_hom_and_tss(np.loadtxt(extrxs_path, skiprows=1)) 
        self.cam_mtrxs = self.extrnsics # camera to world [down, right, backward]
        self.min_t = np.ceil(self.ts[0])
        # cam_mtrxs = self.extrnsics[:,:3] # camera to world [down, right, backward]

        # [down, right, backward] -> [-right, -up, forward] ##########################################################
        # cam_mtrxs = np.stack([cam_mtrxs[:,1, :], cam_mtrxs[:,0, :], cam_mtrxs[:,2, :]], axis=-2) # [right, down, backward]
        # cam_rot, self.coords = cam_mtrxs[:,:3,:3] ,cam_mtrxs[:,:3, 3]
        # cam_rot = np.stack([-cam_rot[:,0, :], cam_rot[:,1,:], -cam_rot[:,2,:]], axis = -2)
        # self.w2cs = cam_rot.transpose(0,2,1)
        ##############################################################################################################

        ######################## blender to opencv ##################################
        # self.coords = cam_mtrxs[:,:3, 3]  # coords in world coordinate
        # w2cs, self.coords = blender_2_opencv(cam_mtrxs)
        # self.w2cs = w2cs[:,:3,:3]
        #############################################################################

        #################### original ########################
        self.w2cs, self.coords = self.extrnsics[:,:3,:3].transpose(0,2,1), self.extrnsics[:,:3,3]
        
        ################### c2w #####################
        # self.w2cs, self.coords = self.extrnsics[:,:3,:3], self.extrnsics[:,:3,3]

        ################### original v2#################
        # cam_mtrxs = np.stack([np.linalg.inv(mtx) for mtx in cam_mtrxs])
        # self.w2cs, ts = cam_mtrxs[:,:3,:3], cam_mtrxs[:, :3, 3]
        # self.coords = np.stack([-t.T@R for t,R in zip(ts, self.w2cs)])

        self.rot_interpolator = Slerp(self.ts, Rotation.from_matrix(self.w2cs))
        self.trans_interpolator = interp1d(x=self.ts, y=self.coords, axis=0, kind="cubic", bounds_error=True)
    

    def interpolate(self, t):
        # t = max(t, self.min_t)
        cond = t < self.min_t
        t[cond] = self.min_t

        return self.trans_interpolator(t), self.rot_interpolator(t).as_matrix()

    def interp_mtx(self, t):
        rot, trans = self.interpolate(t)
        return np.concatenate([rot, trans], axis=-1)

