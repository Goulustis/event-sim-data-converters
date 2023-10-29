from scipy.spatial.transform import Slerp, Rotation
import cv2
import json
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os.path as osp
import glob
from tqdm import tqdm 
import numpy as np
import os
import shutil
from utils import gen_colcam_triggers, read_cameras
import argparse

def create_txt_triggers():
    from utils import gen_colcam_triggers
    img_dir = "/scratch/matthew/projects/synth_datapipeline/synthetic_ev_scene/fine_frames/gamma"
    max_t = int(10*1e6)

    trig_ts = gen_colcam_triggers(img_dir, max_t=max_t, mode="start")

    with open("triggers.txt", "w") as f:
        for t in trig_ts:
            f.write(str(int(t)) + "\n")

    print("done creating triggers")

def reformate_event_h5():
    import h5py
    import numpy as np
    file_path = "/scratch/matthew/projects/synth_datapipeline/synthetic_ev_scene/events.hdf5"
    with h5py.File(file_path, "r") as f:
        xs,ys,ts,ps = [f[e][:] for e in tqdm(list("xytp"), desc="loading h5")]
    
    print("formatting ps")
    cond = ps < 0
    ps[cond] = 0
    ps = ps.astype(np.uint8)

    print("writing h5")
    out_f = "events.h5"
    with h5py.File(out_f, "w") as hf:
        hf.create_dataset('x', data=xs, shape=xs.shape, dtype=xs.dtype)
        hf.create_dataset('y', data=ys, shape=ys.shape, dtype=ys.dtype)
        hf.create_dataset('t', data=ts, shape=ts.shape, dtype=ts.dtype)
        hf.create_dataset('p', data=ps, shape=ps.shape, dtype=np.uint8)

    print("done")

# def evs_to_img(evs:np.ndarray, abs = True):
#     evs = evs.squeeze()
#     if len(evs.shape) == 2:
#         evs = evs[None]

#     frames = np.zeros((*evs.shape, 3), dtype=np.uint8)
#     pos_loc = evs > 0
#     neg_loc = evs < 0

#     if abs:
#         frames[pos_loc, 1] = 255
#         frames[neg_loc, 0] = 255
#     else:
#         to_uint8 = lambda x : (x/x.max() * 255).astype(np.uint8)
#         pos_evs, neg_evs = evs[pos_loc], np.abs(evs[neg_loc])
#         pos_evs, neg_evs = to_uint8(pos_evs), to_uint8(neg_evs)
#         frames[pos_loc, 1] = pos_evs
#         frames[neg_loc, 0] = neg_evs

#     return frames

def evs_to_img(evs:np.ndarray, abs = True):
    evs = evs.squeeze()
    if len(evs.shape) == 2:
        evs = evs[None]

    frames = np.zeros((*evs.shape, 3), dtype=np.uint8)

    for i, ev in tqdm(enumerate(evs), total=len(evs)):
        pos_loc = ev > 0
        neg_loc = ev < 0

        if abs:
            frames[i, pos_loc, 1] = 255
            frames[i, neg_loc, 0] = 255
        else:
            to_uint8 = lambda x : (x/x.max() * 255).astype(np.uint8)
            pos_evs, neg_evs = ev[pos_loc], np.abs(ev[neg_loc])
            pos_evs, neg_evs = to_uint8(pos_evs), to_uint8(neg_evs)
            frames[i, pos_loc, 1] = pos_evs
            frames[i, neg_loc, 0] = neg_evs

    return frames

def check_repeat():
    #import h5py
    import numpy as np
    f = np.load("linked_events.npy", "r")

    xs, ys, ts = f["x"][:500000], f["y"][:500000], f["t"][:500000]


    seen_evs = {}
    prev_t = ts[0]
    for x,y,t in tqdm(zip(xs, ys, ts), total=len(ts)):
        seen_key = f"{x}, {y}, {t}"
        
        if prev_t != t:
            prev_t = t
            del seen_evs
            seen_evs = {}

        if seen_evs.get(seen_key) is None:
            seen_evs[seen_key] = True
        else:
            assert 0


def create_clear_coarse(gap_size=8):
# def create_clear_coarse(gap_size=45):
    st_idx = gap_size//2
    print("creating clear coarse")
    scene_path = "cat_lerp"
    TARG_DIR = osp.join(scene_path, "clear_coarse_frames")
    # SRC_DIRS = [osp.join(scene_path, "fine_frames/gamma"), osp.join(scene_path, "fine_frames/linear")]
    SRC_DIRS = [osp.join(scene_path, "fine_frames/linear")]

    def cp_files(fs, targ_dir):
        os.makedirs(targ_dir, exist_ok=True)
        for i, f in enumerate(fs):
            dst_f = osp.join(targ_dir, f'{str(i).zfill(4)}.png')
            shutil.copy(f, dst_f)
    
    def write_trig(trigs, targ_dir):
        with open(osp.join(targ_dir, "triggers.txt"), "w") as f:
            for t in trigs:
                f.write(str(t) + "\n")
    
    print("done clear coarse")
            
        

    for src_dir in SRC_DIRS:
        ver = osp.basename(src_dir)
        img_fs = np.array(sorted(glob.glob(osp.join(src_dir, "*.png"))))
        mv_idxs = np.array(list(range(len(img_fs))))[st_idx:][::gap_size]
        mv_img_fs = img_fs[mv_idxs]
        targ_dir = osp.join(TARG_DIR, ver)
        cp_files(mv_img_fs, targ_dir)

        triggers = gen_colcam_triggers(src_dir)[mv_idxs]
        write_trig(triggers, targ_dir)



def create_clear_coarse_paramed(src_dir, dst_dir, gap_size=8):
    st_idx = gap_size//2
    def cp_files(fs, targ_dir):
        os.makedirs(targ_dir, exist_ok=True)
        for i, f in enumerate(fs):
            dst_f = osp.join(targ_dir, f'{str(i).zfill(4)}.png')
            shutil.copy(f, dst_f)
    
    def write_trig(trigs, targ_dir):
        with open(osp.join(targ_dir, "triggers.txt"), "w") as f:
            for t in trigs:
                f.write(str(t) + "\n")
    
    print("done clear coarse")

    img_fs = np.array(sorted(glob.glob(osp.join(src_dir, "*.png"))))
    mv_idxs = np.array(list(range(len(img_fs))))[st_idx::gap_size]
    mv_img_fs = img_fs[mv_idxs]
    targ_dir = dst_dir
    cp_files(mv_img_fs, targ_dir)

    triggers = gen_colcam_triggers(src_dir)[mv_idxs]
    write_trig(triggers, targ_dir)


def get_every_n(n = 32):
    SRC_DIR = "synthetic_ev_scene/fine_frames/gamma"
    DST_DIR = "robo_sub32"

    os.makedirs(DST_DIR, exist_ok=True)

    img_fs = sorted(glob.glob(osp.join(SRC_DIR, "*.png")))
    img_fs = img_fs[::n]

    for img_f in img_fs:
        shutil.copy(img_f, DST_DIR)


def make_rel_cam_json():
    intr_json = "synthetic_ev_scene/intrinsics.json"
    with open(intr_json, "r") as f:
        data = json.load(f)
    
    fx = data["focal_length"]
    intr_mtx = np.array([[fx, 0, data["principal_point_x"]],
                          [0, fx,data["principal_point_y"]],
                          [0,  0,                        0]])
    
    dist = np.zeros((5,1))
    rel_json = {"M1": intr_mtx.tolist(),
                "M2": intr_mtx.tolist(),
                "dist1": dist.tolist(),
                "dist2": dist.tolist(),
                "R" : np.eye(3).tolist(),
                "T": np.zeros((3,1)).tolist()}
    
    with open("rel_cam.json", "w") as f:
        json.dump(rel_json, f, indent=4)



def comb_frames(vid_ls, axis=1):
    """
    vid_frames: (list of list of frames (eg. [Vid1, Vid2])); vid1=[img1, img2 ...]
    axis (int): 1 for horizontal, 0 for vertical
    """
    n_frames = min([len(e) for e in vid_ls])
    h, w, c = vid_ls[0][0].shape[-3:]

    new_shape = [n_frames, h, w, c]
    new_shape[axis+1] = new_shape[axis+1]*len(vid_ls)

    res_frames = np.empty(new_shape, dtype=np.uint8)
    for frame_i in tqdm(range(n_frames), desc="combining frames"):
        res_frame = []
        for vid_i in range(len(vid_ls)):
            res_frame.append(vid_ls[vid_i][frame_i])
        
        res_frame = np.concatenate(res_frame, axis=axis)
        res_frames[frame_i] = res_frame
        
    return res_frames

def write_num(frame, number):
    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2

    # Determine the size of the text
    text_size, _ = cv2.getTextSize(str(number), font, font_scale, line_type)

    # Define the position of the text on the frame
    text_position = (10, text_size[1] + 10)  # Adjust position as needed

    # Write the number on the frame
    cv2.putText(frame, str(number), text_position, font, font_scale, font_color, line_type)

    return frame

def create_ev_vid(add_frame_num=True):
    # eimg_f = "/scratch/matthew/projects/e-nerf_synth_datapipeline/cat_lerp_carpet_formatted/ecam_set/eimgs/eimgs_1x.npy"
    eimg_f = "/home/hunter/projects/event-sim-data-converters/formatted_data/cat_simple/ecam_set/eimgs/eimgs_1x.npy"
    print("creating eimg vid")
    ev_vid = evs_to_img(np.load(eimg_f, "r"))
    if add_frame_num:
        for i, frame in enumerate(ev_vid):
            ev_vid[i] = write_num(frame, i)

    print("create done")
    h, w = ev_vid[0].shape[:2]
    # resize = lambda x : cv2.resize(x, (w//2, h//2), interpolation = cv2.INTER_AREA)
    # ev_vid = [resize(e) for e in ev_vid]
    ev_vid = [e for e in ev_vid]
    
    clip = ImageSequenceClip(ev_vid, fps=60)
    clip.write_videofile("adapt_vid.mp4")

    for i, frame in enumerate(ev_vid):
        dst_f = osp.join("ev_frames", str(i).zfill(5) + ".png")
        cv2.imwrite(dst_f, frame)



def create_comb_evid():
    eimg_f1 = "synth_lerp_robo/ecam_set/eimgs/eimgs_1x.npy"
    eimg_f2 = "synth_tex_robo/ecam_set/eimgs/eimgs_1x.npy"

    evid1 = evs_to_img(np.load(eimg_f1))
    evid2 = evs_to_img(np.load(eimg_f2))
    vid = comb_frames([evid1, evid2])
    vid = [e for e in vid]
    clip = ImageSequenceClip(vid, fps=70)
    clip.write_videofile("comb_ev_vid.mp4")



def comb_cat_robo():
    robo_img_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/synth_robo/colcam_set/rgb/1x"
    cat_img_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/cat_plain_formatted/colcam_set/rgb/1x"

    robo_img_fs = sorted(glob.glob(osp.join(robo_img_dir, "*.png")))
    cat_img_fs = sorted(glob.glob(osp.join(cat_img_dir, "*.png")))

    cat_imgs = [cv2.imread(f) for f in tqdm(cat_img_fs, desc="loading cat")]
    h, w = cat_imgs[0].shape[:2]
    new_size = (w, h)
    robo_imgs = [cv2.resize(cv2.imread(f), (w, h)) for f in tqdm(robo_img_fs, desc="loading robo")]

    vid = comb_frames([cat_imgs, robo_imgs])
    vid = [e for e in vid]
    clip = ImageSequenceClip(vid, fps=60)
    clip.write_videofile("cat_robo_rgb.mp4")



def find_comp_orientation(matrices, axis):

    if axis == 1:
        right, up, forward = [matrices[:,i, :] for i in range(3)]
    elif axis == 2:
        right, up, forward = [matrices[:,:, i] for i in range(3)]
    
    idxs = list(range(1, len(matrices) + 1))
    for i in range(8):
        b_num = f'{i:08b}'
        a, b, c = [int(e) for e in b_num[-3:]]
        new_rots = np.stack([right*(-1)**(a), up*(-1)**(b), forward], axis = -1)
        rot_interpolator = Slerp(idxs, Rotation.from_matrix(new_rots))

        if (np.abs((rot_interpolator(idxs).as_matrix() - new_rots).reshape(-1)) < 1e-4).all():
            break
    
    assert (np.abs((rot_interpolator(idxs).as_matrix() - new_rots).reshape(-1)) < 1e-4).all(), "mirror transform!"

    return (a,b,0), new_rots

def conform_to_slerp():
    """
    change camera such that it will conform to slerp
    """
    cam_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/adapt_carpet_slerp/colcam_set/camera"
    cams = read_cameras(cam_dir)

    rots = []
    for cam in cams:
        rots.append(cam["orientation"])
    rots = np.stack(rots)

    (c1,c2,c3), new_rots = find_comp_orientation(rots, 1)
    print(f"orientation found: {c1,c2,c3}")

    for i, new_rot in enumerate(new_rots):
        cams[i]["orientation"] = new_rot.to_list()
    
    cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))
    
    for i, cam_f in enumerate(cam_fs):
        with open(cam_f, "w") as f:
            json.dump(cams[i], f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", help="directory containing images", default="")
    parser.add_argument("--dst_dir", help="directory to save the data", default="")
    parser.add_argument("--gap_size", help="number of frames in between selected frames", default=8)
    args = parser.parse_args()
    create_clear_coarse_paramed(args.src_dir, args.dst_dir, args.gap_size)
    # create_ev_vid()
    # comb_cat_robo()
    # conform_to_slerp()
