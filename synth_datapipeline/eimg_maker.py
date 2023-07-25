import numpy as np
from tqdm import tqdm
import os.path as osp
import h5py
from utils import EventBuffer

def ev_to_img(x, y, p, e_thresh=0.15, img_size = (1080, 1920)):
    """
    input:
        evs (np.array [type (t, x, y, p)]): events such that t in [t_st, t_st + time_delta]
    return:
        event_img (np.array): of shape (h, w)
    """
    e_thresh = 0.15
    h, w = img_size    
    h, w = int(h), int(w)

    pos_p = p==1
    neg_p = p==-1

    e_img = np.zeros((h,w), dtype=np.int16)
    # cnt = np.zeros((h,w), dtype=np.int32)

    # e_img[y[pos_p], x[pos_p]] += e_thresh
    # e_img[y[neg_p], x[neg_p]] -= e_thresh
    # e_img[y[pos_p], x[pos_p]] += 1
    # e_img[y[neg_p], x[neg_p]] -= 1
    np.add.at(e_img, (y, x), p)
    assert np.abs(e_img).max() < np.iinfo(np.int8).max

    return e_img.astype(np.int8)


def create_event_imgs(events: EventBuffer, triggers=None, time_delta=5000, 
                      img_size=(1080, 1920), create_eimgs = True, st_t=0):
    """
    input:
        events (np.array [type (t, x, y, p)]): events
        triggers (np.array [int]): list of trigger time; will generate tight gap if none
        time_delta (int): time in ms, the time gap to create event images
        st_t (int): starting time to accumulate the event images
        create_imgs (bool): actually create the event images, might use this function just to
                            get timestamps and ids

    return:
        eimgs (np.array): list of images with time delta of 50
        eimgs_ts (np.array): list of time at which the event image is accumulated to
        eimgs_ids (np.array): list of embedding ids for each image
        trigger_ids (np.array): list of embedding ids of each trigger time
    """
    if create_eimgs:
        print("creating event images")
    else:
        print("not creating event images, interpolating cameras and creating ids only")



    eimgs = []       # the event images
    eimgs_ts = []    # timestamp of event images
    eimgs_ids = []   # embedding ids for each img
    trig_ids = []    # id at each trigger
    eimgs_ts_end = []

    id_cnt = 0
    with tqdm(total=(len(triggers) - 1)) as pbar:
        for trig_idx in range(1, len(triggers)):
            trig_st, trig_end = triggers[trig_idx - 1], triggers[trig_idx]

            if (events is not None) and create_eimgs:
                curr_t, curr_x, curr_y, curr_p = events.retrieve_data(trig_st, trig_end)
                
            st_t = trig_st
            end_t = trig_st + time_delta
            trig_ids.append(id_cnt)

            while st_t < trig_end:
                if (events is not None) and create_eimgs:
                    cond = (st_t <= curr_t) & (curr_t <= end_t)
                    eimg = ev_to_img(curr_x[cond], curr_y[cond], curr_p[cond], img_size = img_size)
                    eimgs.append(eimg)

                eimgs_ids.append(id_cnt)
                eimgs_ts.append(st_t)
                # eimgs_ts_end.append(end_t)

                # update
                st_t = end_t
                end_t = end_t + time_delta
                end_t = min(end_t, trig_end)
                id_cnt += 1

            pbar.update(1)

    if (events is not None) and create_eimgs:
        return np.stack(eimgs), np.array(eimgs_ts, dtype=np.int32), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32), np.stack(eimgs_ts)
    else:
        return None, np.array(eimgs_ts, dtype=np.int32), np.array(eimgs_ids, dtype=np.int32), np.array(trig_ids, dtype=np.int32), np.stack(eimgs_ts)


def create_eimg_v2(end_ts):
    width = 1920
    height = 1080

    event_file = h5py.File("synthetic_ev_scene/events.hdf5", "r")

    x = event_file["x"]
    y = event_file["y"]
    t = event_file["t"]
    p = event_file["p"]

    N = len(x)
    chunk_size = 2**15

    x_buffer = x[:chunk_size]
    y_buffer = y[:chunk_size]
    t_buffer = t[:chunk_size]
    p_buffer = p[:chunk_size]

    idx = chunk_size
    eimgs = np.zeros((len(end_ts), height, width), dtype=np.int8)

    n_frames = len(end_ts)
    frame_idx = 0
    t_end = end_ts[frame_idx]#1e7 * (frame_idx + 1) / float(n_frames)

    # progress bar
    pbar = tqdm(total=n_frames)

    while (idx < N) and frame_idx < n_frames:
        if len(t_buffer) == 0 or t_buffer[-1] < t_end:
            # read more data
            x_buffer = np.concatenate((x_buffer, x[idx:idx + chunk_size]))
            y_buffer = np.concatenate((y_buffer, y[idx:idx + chunk_size]))
            t_buffer = np.concatenate((t_buffer, t[idx:idx + chunk_size]))
            p_buffer = np.concatenate((p_buffer, p[idx:idx + chunk_size]))

            idx += chunk_size
            continue

        # find the first event that is after the current frame
        last_t_idx = np.searchsorted(t_buffer, t_end, side="right")

        x_frame = x_buffer[:last_t_idx + 1]
        x_buffer = x_buffer[last_t_idx + 1:]

        y_frame = y_buffer[:last_t_idx + 1]
        y_buffer = y_buffer[last_t_idx + 1:]

        t_buffer = t_buffer[last_t_idx + 1:]

        p_frame = p_buffer[:last_t_idx + 1]
        p_buffer = p_buffer[last_t_idx + 1:]

        
        ev_img = np.zeros((height, width), dtype=np.int16)
        np.add.at(ev_img, (y_frame, x_frame), p_frame)

        eimgs[frame_idx] = ev_img
        frame_idx += 1
        # t_end = 1e7 * (frame_idx + 1) / float(n_frames)
        if not (frame_idx < n_frames):
            break

        t_end = end_ts[frame_idx]
        pbar.update(1)

    pbar.close()
    np.save("eimgs.npy",eimgs)
    print("done saving")


if __name__ == "__main__":
    create_eimg_v2(np.load("eimg_end_ts.npy"))

# PLAN:
# 1) use img_times as points for interpolation
# 2) use trigger ids for warp embd id in color dataset