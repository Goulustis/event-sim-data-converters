import cv2
import glob
import os.path as osp
import os
import numpy as np
from tqdm import tqdm

def main():
    src_dir = "rgbs/robo_rgb_linear"
    dst_dir = "rgbs/robo_rgb_linear_4x"
    img_fs = sorted(glob.glob(osp.join(src_dir, "*.png")))

    os.makedirs(dst_dir, exist_ok=True)

    for img_f in tqdm(img_fs, desc="resizing"):
        img = cv2.imread(img_f)
        h, w= tuple(np.array(img.shape)//4)[:2]

        img = cv2.resize(img, (w, h))
        dst_path = osp.join(dst_dir, osp.basename(img_f))
        flag = cv2.imwrite(dst_path, img)
        assert flag, "save failed"


if __name__ == "__main__":
    main()
