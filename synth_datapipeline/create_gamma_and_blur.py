import numpy as np
import os.path as osp
import glob
import os
import argparse
import cv2
from concurrent import futures
from tqdm import tqdm
import matplotlib.pyplot as plt

def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
    """Parallel version of map()."""
    with futures.ThreadPoolExecutor(max_threads) as executor:
        if show_pbar:
            # pylint: disable=g-import-not-at-top
            import tqdm
            results = tqdm.tqdm(
                executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
        else:
            results = executor.map(f, iterable, **kwargs)
        
        return list(results)

def create_blur_imgs(img_dir, dst_dir):
    """
    img_dir (str): directory containing clear pngs used for blurring
    dst_dir (str): directory to save the images
    """
    img_fs = sorted(glob.glob(osp.join(img_dir, "*.png")))
    n_imgs = len(img_fs)

    delta_idx_n = n_imgs//256
    imgs = np.stack(parallel_map(lambda x : cv2.imread(x), img_fs, show_pbar=True, desc="loading imgs"))

    for i in range(256):
        save_f = osp.join(dst_dir, f"{str(i).zfill(4)}.png")
        blur_img = imgs[delta_idx_n*i: delta_idx_n*(i + 1)].mean(axis=0).astype(np.uint8)
        cv2.imwrite(save_f, blur_img)



def u16_to_u8_imgs(img_dir):
    img_fs = glob.glob(osp.join(img_dir, "*.png"))
    parallel_map(lambda f : cv2.imwrite(f, cv2.imread(f)), img_fs, True, desc="u16 to u8")



def create_gamma_imgs(src_dir, dst_dir):
    gamma_f = lambda x : x ** (1/2.4)

    img_fs = sorted(glob.glob(osp.join(src_dir, "*.png")))
    for i, img_f in tqdm(enumerate(img_fs), total=len(img_fs), desc=f"gammafying"):
        dst_f = osp.join(dst_dir, osp.basename(img_f))
        linear = cv2.imread(img_f)
        gamma = (gamma_f(linear/np.iinfo(linear.dtype).max)*255).astype(np.uint8)
        cv2.imwrite(dst_f, gamma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="formated dataset containing ecam_set and colcam_set", default="/home/hunter/projects/ev_sim_converters/formatted_data/cat_fancy")
    parser.add_argument("--src_img_dir", help="generated 4096 imgs", default = "/home/hunter/projects/ev_sim_converters/3D-Graphics-Engine/generated_imgs/cat_fancy_4096")

    args = parser.parse_args()

    clear_linear_src_dir = osp.join(args.dataset_dir, "clear_linear_colcam_set", "rgb", "1x")
    blur_linear_src_dir = osp.join(args.dataset_dir, "blur_linear_colcam_set", "rgb", "1x")
    clear_gamma_dst_dir = osp.join(args.dataset_dir, "clear_gamma_colcam_set", "rgb", "1x")
    blur_gamma_dst_dir = osp.join(args.dataset_dir, "blur_gamma_colcam_set", "rgb", "1x")

    [os.makedirs(dir_path, exist_ok=True) for dir_path in [clear_linear_src_dir, clear_gamma_dst_dir, blur_linear_src_dir, blur_gamma_dst_dir]]

    colcam_set_rgb_dir = osp.join(args.dataset_dir, "colcam_set", "rgb", "1x")
    src_dir = args.src_img_dir if not (args.src_img_dir is None) else colcam_set_rgb_dir

    create_blur_imgs(src_dir, blur_linear_src_dir)

    create_gamma_imgs(clear_linear_src_dir, clear_gamma_dst_dir)
    create_gamma_imgs(blur_linear_src_dir, blur_gamma_dst_dir)

    u16_to_u8_imgs(colcam_set_rgb_dir)
    u16_to_u8_imgs(clear_linear_src_dir)