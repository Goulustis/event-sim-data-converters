import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm
import glob
import os.path as osp
import matplotlib.pyplot as plt
from concurrent import futures

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

def eimg_to_img(evs):

    rgb = np.zeros(list(evs.shape) + [3])

    neg_cond = evs < 0
    pos_cond = evs > 0

    rgb[neg_cond, 0] = np.abs(evs[neg_cond])
    rgb[pos_cond, 1] = evs[pos_cond]

    rgb = rgb/rgb.max()
    rgb = np.floor(rgb*255).astype(np.uint8)
    return rgb

def main():
    eimg_f = "ShakeCarpet1_formatted/ecam_set/eimgs/eimgs_1x.npy"
    eimgs = np.load(eimg_f)


    frames = []
    for eimg in tqdm(eimgs):
        rgb_frame = eimg_to_img(eimg)
        frames.append(rgb_frame)
    
    vid = ImageSequenceClip(frames, fps=25)
    vid.write_videofile("ev_vid.mp4")


def to_vid():
    img_dir = "synthetic_ev_scene/ev_frames"
    img_fs = sorted(glob.glob(osp.join(img_dir, "*.png")))

    vid_frames = []
    for img_f in tqdm(img_fs):
       img = (plt.imread(img_f)*255).astype(np.uint8)
       vid_frames.append(img)

    vid = ImageSequenceClip(vid_frames, fps=32)
    vid.write_videofile("ori_vid.mp4")


if __name__ == "__main__":
    main()