import os.path as osp
import glob
import os
import numpy as np
from tqdm import tqdm
import shutil

def main():
    rgb_dir = "generated_data/adapt_carpet/intensity_imgs"
    trig_f = osp.join(rgb_dir, "triggers.txt")
    targ_dir = osp.join("/home/hunter/projects/colmap_workdir", osp.basename(osp.dirname(rgb_dir)) + "_recons")
    targ_rgb_dir = osp.join(targ_dir, "images")
    os.makedirs(targ_rgb_dir, exist_ok=True)
    gap = 5

    triggers = np.loadtxt(trig_f)
    img_fs = sorted(glob.glob(osp.join(rgb_dir, "*.png")))

    assert len(triggers) == len(img_fs), "triggers and imgs are not same length!"

    triggers = triggers[::gap]
    img_fs = img_fs[::gap]

    for i, img_f in tqdm(enumerate(img_fs),total=len(img_fs), desc="copying imgs"):
        targ_img_f = osp.join(targ_rgb_dir, str(i).zfill(6) + ".png")
        shutil.copy(img_f, targ_img_f)
    
    with open(osp.join(targ_dir, "triggers.txt"), "w") as f:
        for t in tqdm(triggers, desc="writing trigs"):
            f.write(str(t) + "\n")


if __name__ == "__main__":
    main()