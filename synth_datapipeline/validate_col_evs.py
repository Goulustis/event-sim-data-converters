import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import glob
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import json
from tqdm import tqdm
import os

def get_ts(cam_fs):
    ts = []

    for cam_f in cam_fs:
        with open(cam_f, "r") as f:
            data = json.load(f)
            ts.append(int(data["t"]))

    return np.array(ts)

def load_all_imgs(colcam_set_f):
    img_fs = sorted(glob.glob(osp.join(colcam_set_f, "rgb", "1x", "*.png")))

    imgs = []
    for img_f in tqdm(img_fs, desc="loading rgb imgs"):
        # imgs.append(plt.imread(img_f))
        imgs.append(np.stack([plt.imread(img_f)]*3, axis=-1))
    
    return imgs


def ev_imify(eimgs):

    ev_imgs = np.zeros((*eimgs.shape, 3))

    for i, eimg in tqdm(enumerate(eimgs), total=len(ev_imgs), desc="imifying events"):
        pos = eimg >0 
        neg = eimg < 0
        ev_imgs[i, neg, 0] = 1
        ev_imgs[i, pos, 1] = 1
    
    return ev_imgs

def make_vid(rgb_imgs, ev_imgs, make_cache=True):
    dst_f = "evs_rgb.mp4"

    frames = []
    for rgb, ev_img in tqdm(zip(rgb_imgs, ev_imgs), total=len(rgb_imgs), desc="making vid"):
        frames.append((np.concatenate([rgb, ev_img], axis=1)*255).astype(np.uint8))
    
    clip = ImageSequenceClip(frames, fps=15)
    clip.write_videofile(dst_f)

    if make_cache:
        create_cache(frames)


def create_cache(frames):
    os.makedirs("val_evs_rgb_cache", exist_ok=True)
    for i, img in tqdm(enumerate(frames), desc="making cache", total=len(frames)):
        plt.imsave(osp.join("val_evs_rgb_cache", str(i).zfill(4) + ".png"), img)

def main():
    ecam_set_f = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/synth_robo/ecam_set"
    colcam_set_f = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/synth_robo/blur_colcam_set"

    eimg_cam_fs = sorted(glob.glob(osp.join(ecam_set_f, "camera", "*.json")))
    col_cam_fs = sorted(glob.glob(osp.join(colcam_set_f, "camera", "*.json")))

    eimg_ts = get_ts(eimg_cam_fs)
    col_ts = get_ts(col_cam_fs)

    diffs = np.abs(col_ts[..., None] - eimg_ts[None])
    min_idx = diffs.argmin(axis=1)

    rgb_imgs = load_all_imgs(colcam_set_f)

    eimg_f = osp.join(ecam_set_f, "eimgs", "eimgs_1x.npy")
    viz_eimgs = np.load(eimg_f, "r")[min_idx]
    ev_imgs = ev_imify(viz_eimgs)
    
    make_vid(rgb_imgs, ev_imgs)



if __name__ == "__main__":
    main()


