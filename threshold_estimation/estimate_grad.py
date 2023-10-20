import numpy as np
import os.path as osp
import os
import h5py
# from threshold_estimation.est_utils import read_timestamp_txt, EventBuffer
from est_utils import read_timestamp, EventBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import cv2


import torch

EPS = 1e-5
def estimate_threshold(img1, img2, events, n_step = 5000):
    """
    img1, img2 = image of shape (h,w)
    """
    ### initialize ###
    ev_img = events.astype(np.float32)
    

    img1, img2 = torch.from_numpy(img1).cuda(), torch.from_numpy(img2).cuda()
    ev_img = torch.from_numpy(ev_img).cuda()

    thresh_est = torch.rand(img1.shape).cuda() * 1e-3
    thresh_est.requires_grad = True

    optim = torch.optim.Adam([thresh_est], lr = 5e-4)

    Is = torch.log(img1 + EPS)
    targ = torch.log(img2 + EPS)
    # for step in tqdm(range(n_step)):
    for step in range(n_step):

        Id = thresh_est * ev_img + Is
        optim.zero_grad()
        error = ((Id - targ)**2).mean()
        error.backward()
        optim.step()

    thresh_est = thresh_est.detach()
    thresh_est[torch.abs(thresh_est) < 1e-2] = 0   
    print((torch.abs(thresh_est).sum()/(thresh_est != 0).sum()).item() * 2.44)  ## for sanity check
    ev_img = ev_img.cpu().numpy().astype(np.float16)
    return thresh_est.cpu().numpy().astype(np.float16) * ev_img

        



def prep_img(img):
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.max() > 1:
        img = img / np.iinfo(img.dtype).max
    
    return img




def main():
    # event_f = "/ubc/cs/research/kmyi/matthew/backup_copy/synth_datapipeline/synthetic_ev_scene_tex/events.hdf5"
    # time_f = "/ubc/cs/research/kmyi/matthew/backup_copy/synth_datapipeline/synthetic_ev_scene_tex/coarse_frames/gamma/timestamp.txt"
    # img_dir = "/ubc/cs/research/kmyi/matthew/backup_copy/synth_datapipeline/synthetic_ev_scene_tex/fine_frames/linear"
    # dst_f = "/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/synth_tex_robo/est_ecam_set/eimgs/eimgs_1x.npy"

    dataset = "cat_fancy"
    event_f = f"/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/{dataset}/ecam_set/eimgs/eimgs_1x.npy"
    time_f = f"/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/{dataset}/blur_gamma_colcam_set/camera"
    img_dir = f"/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/{dataset}/blur_gamma_colcam_set/rgb/1x"
    dst_f = f"/ubc/cs/research/kmyi/matthew/projects/ed-nerf/data/{dataset}/est_ecam_set/eimgs/eimgs_1x.npy"


    ev_buffer = np.load(event_f, "r")
    img_fs = sorted(glob.glob(osp.join(img_dir ,"*.png")))


    e_threshes = np.zeros((len(img_fs)-1, *cv2.imread(img_fs[0]).shape[:2]), dtype=np.float16)
    for i in tqdm(range(len(img_fs)-1)):
        img1, img2 = cv2.imread(img_fs[i], cv2.IMREAD_UNCHANGED), cv2.imread(img_fs[i+1], cv2.IMREAD_UNCHANGED)
        # img1, img2 = cv2.imread(img_fs[i]), cv2.imread(img_fs[i+1])
        img1, img2 = prep_img(img1), prep_img(img2)


        delta_idx = int(len(ev_buffer)//(len(img_fs) - 1))
        events = ev_buffer[i*delta_idx : (i+1)*delta_idx].sum(axis=0)
        thresh_est = estimate_threshold(img1, img2, events)
        e_threshes[i] = thresh_est
    
    np.save(dst_f, e_threshes)


if __name__ == "__main__":
    main()
