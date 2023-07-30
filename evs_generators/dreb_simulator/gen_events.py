import numpy as np
import h5py
import tqdm
import glob
import os.path as osp
import cv2 
import argparse

EPS = 1e-7

def gen_events(src_dir, targ_f, C=0.2):
    print("running")
    fs = sorted(glob.glob(osp.join(src_dir, "*.png")))
    n_frames = len(fs)
    # frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for f in tqdm.tqdm(fs, desc="loading images")]
    frames = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in tqdm.tqdm(fs, desc="loading images")]
    event_file = h5py.File(targ_f, "w")

    x_out = event_file.create_dataset("x", (0, ), dtype=np.uint16, maxshape=(None, ))
    y_out = event_file.create_dataset("y", (0, ), dtype=np.uint16, maxshape=(None, ))
    p_out = event_file.create_dataset("p", (0, ), dtype=np.int16, maxshape=(None, ))
    t_out = event_file.create_dataset("t", (0, ), dtype=np.int64, maxshape=(None, ))

    idx = 0

    prev_frame_intensity = np.clip(frames[0], EPS, 1e6)
    # prev_event_log_intensity = np.log(prev_frame_intensity)
    prev_event_log_intensity = prev_frame_intensity
    for i in tqdm.trange(1, len(frames)):
        t_start = (i - 1) / float(n_frames)
        t_end = i / float(n_frames)

        current_intensity = np.clip(frames[i], EPS, 1e6)
        current_intensity = np.where(np.isfinite(current_intensity), current_intensity, prev_frame_intensity)
        delta_intensity = current_intensity - prev_frame_intensity
        # current_log_intensity = np.log(current_intensity)
        current_log_intensity = current_intensity

        delta_log_intensity = current_log_intensity - prev_event_log_intensity
        discrete_magnitude = np.floor(np.abs(delta_log_intensity) / C)
        discrete_magnitude = discrete_magnitude.astype(np.int32)

        event_mask = (discrete_magnitude > 0) & (np.abs(delta_intensity) > EPS)

        polarity = np.sign(delta_log_intensity)
        event_delta_log_intensity = polarity * discrete_magnitude * C

        events_x = []
        events_y = []
        events_p = []
        events_t = []

        xy = np.argwhere(event_mask)

        for y, x in xy:
            event_log_intensities = prev_event_log_intensity[y, x] + np.linspace(0.0, event_delta_log_intensity[y, x], discrete_magnitude[y, x] + 1)[1:]
            event_intensities = np.exp(event_log_intensities)
            rel_event_times = (event_intensities - prev_frame_intensity[y, x]) / (current_intensity[y, x] - prev_frame_intensity[y, x])
            timestamps = t_start + rel_event_times * (t_end - t_start)

            for j in range(discrete_magnitude[y, x]):
                # frame_events.append((x, y, polarity[y, x], timestamps[j]))
                events_x.append(x)
                events_y.append(y)
                events_p.append(polarity[y, x])
                events_t.append(timestamps[j])

        perm = np.argsort(events_t)
        events_x = np.array(events_x)[perm]
        events_y = np.array(events_y)[perm]
        events_p = np.array(events_p)[perm]
        events_t = np.array(events_t)[perm]
        n = len(events_x)

        x_out.resize((idx + n, ))
        y_out.resize((idx + n, ))
        p_out.resize((idx + n, ))
        t_out.resize((idx + n, ))

        x_out[idx:idx + n] = events_x.astype(np.uint16)
        y_out[idx:idx + n] = events_y.astype(np.uint16)
        p_out[idx:idx + n] = events_p.astype(np.int16)
        t_out[idx:idx + n] = np.round(events_t * 1e7).astype(np.int64)

        idx += n

        prev_event_log_intensity += event_delta_log_intensity
        prev_frame_intensity = current_intensity

        if i % 10 == 0:
            event_file.flush()

    event_file.flush()
    event_file.close()


if __name__ == "__main__":
    parser = argparse.parser()
    parser.add_argument("--frame_dir", default = "/scratch/matthew/projects/synth_datapipeline/synthetic_ev_scene/fine_frames/gamma")
    parser.add_argument("--targ_f", default = "events.hdf5")
    args =  parser.parse_args()

    # frame_dir = "/scratch/matthew/projects/synth_datapipeline/synthetic_ev_scene/fine_frames/gamma"
    # targ_f = "events.hdf5"
    frame_dir = args.frame_dir
    targ_f = args.targ_f
    gen_events(frame_dir, targ_f)