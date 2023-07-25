import json
import numpy as np
import os
import os.path as osp
from utils import read_evs_h5, read_intrinsics, read_intrinsics_from_cam_f
from tqdm import tqdm
import shutil
import cv2
import glob
from synthetic_ev_scene.camera_spline import CameraSpline, CameraSpline_enerf
from scipy.spatial.transform import Rotation



ROBO_CENTER = [-8.30339975, 18.90998993, 58.92599333]  # center of robot scene

# (x, y, ts_ns, p)
# def format_events(src_dir, dst_dir, triggers):
#     ev_f = osp.join(src_dir,"ecam_set", "events.hdf5")
#     dst_ev_dir = osp.join(dst_dir, "events")
#     os.makedirs(dst_ev_dir, exist_ok=True)

#     evs_ori = read_evs_h5(ev_f)
#     evs = np.stack([evs_ori['x'],evs_ori['y'], evs_ori['t'], evs_ori['p']], axis = -1)
#     evs_batches = []

#     ts = evs_ori["t"]
#     for i in tqdm(range(len(triggers) - 1), desc='batching evs'):
#         st_t, end_t = triggers[i], triggers[i+1]
#         cond = (ts >= st_t) & (ts <= end_t)
#         evs_batches.append(evs[cond])
    
#     for i, ev_batch in tqdm(enumerate(evs_batches), total = len(evs_batches), desc="saving evs"):
#         dst_f = osp.join(dst_ev_dir, f"{str(i).zfill(4)}.npy")
#         np.save(dst_f, ev_batch)

def format_events(src_dir, dst_dir, triggers, ev_f = None):
    # triggers = triggers * 1000   # robo
    triggers = triggers * 1e6  # esim
    ev_f = osp.join(src_dir,"ecam_set", "events.hdf5") if ev_f is None else ev_f
    dst_ev_dir = osp.join(dst_dir, "events")
    os.makedirs(dst_ev_dir, exist_ok=True)

    evs_ori = read_evs_h5(ev_f)
    # evs = np.stack([evs_ori['x'],evs_ori['y'], evs_ori['t']*1000, evs_ori['p']], axis = -1) #robo
    evs = np.stack([evs_ori['x'],evs_ori['y'], evs_ori['t'], evs_ori['p']], axis = -1)  # esim
    cond = evs[:, 3] == 0
    evs[cond, 3] = -1
    evs_batches = []

    # triggers[-1] = evs[-1,2] #esim

    ts = evs_ori["t"]*1000
    for i in tqdm(range(len(triggers)-1), desc='batching evs'):
        st_t, end_t = triggers[i], triggers[i+1]
        cond = (ts >= st_t) & (ts <= end_t)
        evs_batches.append(evs[cond])
    
    evs_batches.append(evs[triggers[-1] <= ts])
    
    
    for i, ev_batch in tqdm(enumerate(evs_batches), total = len(evs_batches), desc="saving evs"):
        dst_f = osp.join(dst_ev_dir, f"{str(i).zfill(4)}.npy")
        np.save(dst_f, ev_batch)


def resize_and_save_img_dir(src_dir, dst_dir, scale=1):
    os.makedirs(dst_dir, exist_ok=True)
    for img_f in tqdm(sorted(glob.glob(osp.join(src_dir, "*.png"))), desc=f"resizing by {scale}x"):
            img, dest_path = cv2.imread(img_f), osp.join(dst_dir, osp.basename(img_f))
            h, w = img.shape[:2]
            flag = cv2.imwrite(dest_path, cv2.resize(img, (w//scale, h//scale)))
            assert flag, f"file saving failed: {dest_path}"



def format_clear_rgbs(rgb_dir, dst_dir, src_dir, scale=1):
    rgb_dst_dir = osp.join(dst_dir, "images")
    trigger_f = osp.join(rgb_dst_dir, "image_stamps_ns.txt")

    triggers = load_triggers(src_dir)

    img_f = glob.glob(osp.join(rgb_dir, "*.png"))[0]
    img_size = cv2.imread(img_f).shape[:2][::-1]

    rgb_dst_shape_dir = osp.join(dst_dir, f"images_{img_size[0]//scale}x{img_size[1]//scale}")

    if scale == 1:
        if osp.exists(rgb_dst_shape_dir):
            shutil.rmtree(rgb_dst_shape_dir)
        shutil.copytree(rgb_dir, rgb_dst_shape_dir)
        shutil.copytree(rgb_dir, rgb_dst_dir, dirs_exist_ok=True)
    else:
        resize_and_save_img_dir(rgb_dir, rgb_dst_shape_dir, scale)
        resize_and_save_img_dir(rgb_dir, rgb_dst_dir, scale)
        # for img_f in tqdm(sorted(glob.glob(osp.join(rgb_dir, "*.png"))), desc=f"resizing to {scale}"):
        #     img, dest_path = cv2.imread(img_f), osp.join(rgb_dst_shape_dir, osp.basename(img_f))
        #     cv2.imwrite(dest_path, img)
    print(f"saved to {rgb_dst_dir} \n         {rgb_dst_shape_dir}")

    with open(trigger_f, "w") as f:
        for t in triggers:
            f.write(str(t * 1000) + "\n")

    shutil.copy(trigger_f, osp.join(rgb_dst_shape_dir, "image_stamps_ns.txt"))

def format_blur_rgbs(blur_rgb_dir, dst_dir, src_dir, scale=1):
    blur_rgb_dst_dir = osp.join(dst_dir, "images_corrupted")
    triggers = load_triggers(src_dir)

    if scale == 1:
        if osp.exists(blur_rgb_dst_dir):
            shutil.rmtree(blur_rgb_dst_dir)
        shutil.copytree(blur_rgb_dir, blur_rgb_dst_dir)
    else:
        resize_and_save_img_dir(blur_rgb_dir, blur_rgb_dst_dir, scale)
    print(f"saved to {blur_rgb_dst_dir}")

    trigger_f = osp.join(blur_rgb_dst_dir, "image_stamps_ns.txt")
    with open(trigger_f, "w") as f:
        for t in triggers:
            f.write(f"{t*1000} \n")


def load_triggers(src_dir):
    colcam_dir = osp.join(src_dir, "colcam_set", "camera")
    cam_fs = sorted(glob.glob(osp.join(colcam_dir, "*.json")))
    ts = []
    for i, cam_f in enumerate(cam_fs):
        with open(cam_f, "r") as f:
            data = json.load(f)
            ts.append(float(data["t"]))

    return np.array(ts) 


# stamps in nanoseconds, px, py, pz, qx, qy, qz, qw
def create_poses_all_txt(dst_dir, src_dir):
    cx, cy, cz = ROBO_CENTER
    poses_f = osp.join(dst_dir, "poses_all.txt")
    camspline = CameraSpline_enerf(src_dir)
    # triggers = np.concatenate([gen_colcam_triggers(n_frames=15000, mode="start"), np.array([int(10*1e6)])])
    triggers = camspline.triggers()
    Ts, Rs = camspline.interpolate(triggers, model="enerf")
    # Ts, Rs = camspline.interpolate(triggers)
    


    with open(poses_f, "w") as f:
        f.write("# stamps in nanoseconds, px, py, pz, qx, qy, qz, qw \n")
        for t, trans, R in tqdm(zip(triggers, Ts, Rs), total=len(Ts), desc="writing poses"):
            x, y ,z = trans
            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            # to_write = f"{t*1000} {x} {y} {z} {qx} {qy} {qz} {qw} \n"
            to_write = f"{t*1000} {x} {y} {z+0.9} {qx} {qy} {qz} {qw} \n"  # [IMPORTANT]: REMOVE THE + 0.3 later
            f.write(to_write)

    print(f"saved to {poses_f}")


def get_min_max_depth(scene_f):
    with open(scene_f, "r") as f:
        data = json.load(f)

    scale = data["scale"]
    # return data["near"]/scale, data["far"]/scale
    return 0.15, 40

# Nx17, first 15 = c2w + [image height, image width, focal length], last 2 = min, max depth
def create_poses_bounds(rgb_dir, intrinsic_f, scene_f, dst_dir, src_dir, scale=1):
    dst_f = osp.join(dst_dir, "poses_bounds.npy")
    # triggers = gen_colcam_triggers(rgb_dir)
    # triggers = load_triggers(src_dir)
    depth_min, depth_max = get_min_max_depth(scene_f)
    camspline = CameraSpline_enerf(src_dir)
    triggers = camspline.triggers()
    Trans, Rs = camspline.interpolate(triggers, model="enerf")
    Trans, Rs = camspline.interpolate(triggers)

    # Trans = Trans - np.array(ROBO_CENTER)
    Trans = Trans[..., None]

    # intrxs = read_intrinsics(intrinsic_f, scale)
    intrxs = read_intrinsics_from_cam_f(intrinsic_f, scale)
    focal = intrxs[0,0]
    cx, cy = intrxs[:2,2].astype(int)
    
    H, W = cy*2, cx*2

    hwf = np.array([H, W, focal])[..., None]
    depth_vec = np.array([depth_min, depth_max])    

    poses_bounds = []
    
    for T, R in tqdm(zip(Trans, Rs), desc="creating poses bounds"):
        comb_R = np.concatenate([R,T,hwf], axis=1)
        poses_bound = np.concatenate([comb_R.reshape(-1), depth_vec])
        poses_bounds.append(poses_bound)
    
    poses_bounds = np.stack(poses_bounds)
    np.save(dst_f, poses_bounds)
    print(f"saved to {dst_f}")

# def main():
#     src_dir = "synth_robo"
#     dst_dir = "enerf_robo_gamma_4x"
#     rgb_dir = "synthetic_ev_scene/clear_coarse_frames/linear"
#     blur_rgb_dir = "synthetic_ev_scene/coarse_frames/linear"
#     intrx_f = "synthetic_ev_scene/intrinsics.json"
#     scene_f = "synth_robo/colcam_set/scene.json"
#     ev_f = "synthetic_ev_scene/robo_events_gamma_4x.hdf5"
#     scale = 1  # dim to reduce by from original images, (eg. scale=2, new_img_size = (H//2, W//2), focal length// 2)

#     os.makedirs(dst_dir, exist_ok=True)

#     triggers = np.concatenate([np.array([0]), gen_colcam_triggers(rgb_dir)])
#     create_poses_all_txt(dst_dir)
#     create_poses_bounds(rgb_dir, intrx_f, scene_f, dst_dir, scale=scale)
#     format_clear_rgbs(rgb_dir, dst_dir, scale=scale)
#     format_blur_rgbs(blur_rgb_dir, dst_dir, scale)
#     format_events(src_dir, dst_dir, triggers, ev_f=ev_f)

def load_triggers_from_src_dir(src_dir):
    colcam_dir = osp.join(src_dir, "colcam_set", "camera")
    cam_fs = sorted(glob.glob(osp.join(colcam_dir, "*.json")))

    ts = []
    for cam_f in cam_fs:
        with open(cam_f, "r") as f:
            ts.append(float(json.load(f)['t']))

    return np.array(ts)

def main():
    src_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_formatted"
    dst_dir = "/scratch/matthew/projects/enerf/data/enerf_adapt_carpet"
    rgb_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_formatted/colcam_set/rgb/1x"
    blur_rgb_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_formatted/colcam_set/rgb/1x"
    # intrx_f = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_formatted/colcam_set/intrinsics.json"
    intrx_f = sorted(glob.glob(osp.join(src_dir, "colcam_set", "camera", "*.json")))[0]
    scene_f = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_formatted/colcam_set/scene.json"
    ev_f = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_formatted/ecam_set/events.h5"
    scale = 1  # dim to reduce by from original images, (eg. scale=2, new_img_size = (H//2, W//2), focal length// 2)

    # src_dir = "synth_robo"
    # dst_dir = "/scratch/matthew/projects/enerf/data/enerf_robo_v4"
    # rgb_dir = "synthetic_ev_scene/clear_coarse_frames/linear"
    # blur_rgb_dir = "synthetic_ev_scene/coarse_frames/linear"
    # intrx_f = "synthetic_ev_scene/intrinsics.json"
    # scene_f = "synth_robo/colcam_set/scene.json"
    # ev_f = "synthetic_ev_scene/robo_events_gamma_4x.hdf5"
    # scale = 1  # dim to reduce by from original images, (eg. scale=2, new_img_size = (H//2, W//2), focal length// 2)

    os.makedirs(dst_dir, exist_ok=True)

    # triggers = np.concatenate([np.array([0]), gen_colcam_triggers(rgb_dir)])
    triggers = load_triggers_from_src_dir(src_dir)
    create_poses_all_txt(dst_dir, src_dir)
    create_poses_bounds(rgb_dir, intrx_f, scene_f, dst_dir, src_dir, scale=scale)
    format_clear_rgbs(rgb_dir, dst_dir,src_dir=src_dir, scale=scale)
    format_blur_rgbs(blur_rgb_dir, dst_dir,src_dir=src_dir, scale=scale)
    format_events(src_dir, dst_dir, triggers, ev_f=ev_f)


def gen_carpet_evs():
    dst_dir = "tmp_dir"
    trigger_f = "/scratch/matthew/projects/e-nerf_synth_datapipeline/ori_data/ShakeCarpet1/images/image_stamps_ns.txt"
    triggers = np.loadtxt(trigger_f, delimiter=",", skiprows=1)

    ev_f = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/ShakeCarpet1_formatted/ecam_set/events.h5"
    format_events(None, dst_dir, triggers, ev_f=ev_f)


if __name__ == "__main__":
    main()