import matplotlib.pyplot as plt
import esim_torch
from utils import gen_colcam_triggers
import glob
import os.path as osp
import os
import cv2
import h5py
from concurrent import futures
import torch
import numpy as np


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

def to_log(img):
    return np.log(img/255 + 1e-4)

@torch.no_grad()
def main():
    vid_dir = "ori_imgs_1x"
    device = "cuda"
    img_fs = sorted(glob.glob(osp.join(vid_dir, "*.png"))[:30])
    img_ts = gen_colcam_triggers(vid_dir)[:30]

    imgs = np.stack(parallel_map(lambda x : cv2.imread(x, cv2.IMREAD_GRAYSCALE), img_fs))
    log_imgs = torch.from_numpy(np.log(imgs/255 + 1e-4)).to(device).float()
    img_ts = torch.from_numpy(img_ts).to(device)


    esim = esim_torch.ESIM()
    events = esim.forward(log_imgs, timestamps=img_ts)

    image = imgs[0]
    
    first_few_events = {k: v[:100000].cpu().numpy() for k,v in events.items()}
    image_color = np.stack([image,image,image],-1)
    image_color[first_few_events['y'], first_few_events['x'], :] = 0
    image_color[first_few_events['y'], first_few_events['x'], first_few_events['p']] = 255

    plt.imshow(image_color)
    plt.show()


if __name__ == "__main__":
    main()
