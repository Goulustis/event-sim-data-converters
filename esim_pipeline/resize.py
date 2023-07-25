import cv2
import glob
import os.path as osp
import os
import numpy as np
from tqdm import tqdm

def main():
    img_fs = sorted(glob.glob("gamma/*.png"))

    os.makedirs("ori_imgs_4x", exist_ok=True)

    for img_f in tqdm(img_fs):
        img = cv2.imread(img_f)
        h, w= tuple(np.array(img.shape)//4)[:2]

        img = cv2.resize(img, (w, h))
        dst_path = osp.join("ori_imgs_4x", osp.basename(img_f))
        flag = cv2.imwrite(dst_path, img)
        assert flag, "save failed"


if __name__ == "__main__":
    main()
