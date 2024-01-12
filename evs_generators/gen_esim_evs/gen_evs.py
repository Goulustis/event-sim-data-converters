import esim_torch
from utils import gen_colcam_triggers
import glob
import os.path as osp
import cv2
import h5py
from concurrent import futures
import torch
import numpy as np
from tqdm import tqdm
import argparse


dtype_dic = {"x": np.dtype("u2"),
             "y": np.dtype("u2"),
             "p": np.dtype("i1"),
             "t": np.dtype("u8")}

def p_val_map(x):
    cond = x < 0
    x[cond] = 0
    return x


id_fn = lambda x : x
val_map = {"x": id_fn,
           "y": id_fn,
           "t": id_fn,
           "p": id_fn}

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


def batchify(imgs, img_ts, batch_size = 128):
    
    st_idx = 0
    end_idx = batch_size
    
    data = []
    while st_idx <= len(img_ts):
        data.append((imgs[st_idx:end_idx], img_ts[st_idx:end_idx]))
        st_idx = end_idx
        end_idx = end_idx + batch_size

    return data


def append_data(batch, data_dic):

    for k, v in batch.items():
        data = val_map[k](v.cpu().numpy()).astype(dtype_dic[k])
        if data_dic.get(k) is None:
            # data_dic[k] = data
            data_dic[k] = [data]
        else:
            # data_dic[k] = np.concatenate([data_dic[k], data])
            data_dic[k].append(data)

    return data_dic

def write_data(out_dic, data_dic):
    for k, v in data_dic.items():
        v = (v.cpu().numpy()).astype(dtype_dic[k])
        curr_sz = len(out_dic[k])
        out_dic[k].resize((curr_sz + len(v), ))
        out_dic[k][curr_sz: curr_sz + len(v)] = v

        if k == "t":
            assert np.all(v >= 0)
    
    

@torch.no_grad()
def main(frame_dir, targ_f, ev_thresh = 0.2, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    # vid_dir = "rgbs/hosp_carpet"
    # device = "cuda"
    # out_f = "hosp_carpet_events.hdf5"
    img_fs = sorted(glob.glob(osp.join(frame_dir, "*.png")))
    # img_ts = gen_colcam_triggers(frame_dir, scene_mode="robo")
    img_ts = np.loadtxt("/home/hunter/projects/ev_sim_converters/3D-Graphics-Engine/camera_data/triggers2048.txt")
    print(f"using total of {len(img_fs)} to generate events")

    event_file = h5py.File(targ_f, "w")
    out_dic = {}
    for k, v in dtype_dic.items():
        out_dic[k] = event_file.create_dataset(k, (0,), dtype=v, maxshape=(None,))

    if len(cv2.imread(img_fs[0], cv2.IMREAD_UNCHANGED).shape) == 3:
        read_fn = lambda x : cv2.cvtColor(cv2.imread(x, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
    else:
        read_fn = lambda x : cv2.imread(x, cv2.IMREAD_UNCHANGED)
    imgs = np.stack(parallel_map(read_fn, img_fs, show_pbar=True, desc="loading imgs"))

    # esim = esim_torch.ESIM(contrast_threshold_neg=ev_thresh, contrast_threshold_pos=ev_thresh, refractory_period_ns=100)
    esim = esim_torch.ESIM(contrast_threshold_neg=ev_thresh, contrast_threshold_pos=ev_thresh)
    # data_iter = batchify(imgs, img_ts, batch_size=16)
    data_iter = batchify(imgs, img_ts, batch_size=32)

    norm_factor = np.iinfo(imgs.dtype).max
    for batch in tqdm(data_iter, desc="generating"):
        b_imgs, b_ts = batch
        log_imgs, b_ts = torch.from_numpy(np.log(np.clip(b_imgs/norm_factor, 1.2e-7, None))).float().to(device), torch.from_numpy(b_ts.astype(int)).to(device)
        evs = esim.forward(log_imgs, b_ts)

        if evs is None:
            continue

        # append_data(evs, data_dic)
        write_data(out_dic, evs)


    # with h5py.File(out_f, "w") as hf:
    #     for k, v in data_dic.items():
    #         data_dic[k] = None
    #         v = np.concatenate(v)
    #         hf.create_dataset(k, data=v, shape=v.shape)
    #         del v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", type=str, default="/home/hunter/projects/ev_sim_converters/3D-Graphics-Engine/generated_imgs/half_checker_2048")
    # parser.add_argument("--frame_dir", type=str, default="/home/hunter/projects/ev_sim_converters/3D-Graphics-Engine/generated_imgs/synth_tex_robo_2048")
    parser.add_argument("--targ_f", type=str, default="re_cat_fancy_evs.hdf5")
    parser.add_argument("--ev_thresh",type=float, default=0.2)
    args = parser.parse_args()

    main(args.frame_dir, args.targ_f, args.ev_thresh)
