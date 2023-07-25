from utils import read_intrinsics, make_camera, EventBuffer, read_intrinsics_json
from camera_spline import CameraSpline
from eimg_maker import create_event_imgs
import os
import os.path as osp
import numpy as np
import json
import shutil

def save_eimgs(eimgs, targ_dir):
    if eimgs is None:
        return 

    eimgs_dir = osp.join(targ_dir, "eimgs")
    os.makedirs(eimgs_dir, exist_ok=True)
    np.save(osp.join(eimgs_dir, "eimgs_1x.npy"), eimgs)
    del eimgs

def create_and_write_camera_extrinsics(extrinsic_dir, orientation, poses, triggers, intr_mtx, ret_cam=False):
    """
    create the extrinsics and save it
    """
    os.makedirs(extrinsic_dir, exist_ok=True)
    cameras = []
    for i, (R, pose ,t) in enumerate(zip(orientation, poses, triggers)):
        targ_cam_path = osp.join(extrinsic_dir, str(i).zfill(6) + ".json")
        camera = make_camera(R, pose, intr_mtx)
        cameras.append(camera)
        print("saving to", targ_cam_path)
        cam_json = camera.to_json()
        cam_json["t"] = int(t)
        with open(targ_cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)

    if ret_cam:
        return cameras

def write_metadata(eimgs_ids, eimgs_ts, targ_dir):
    """
    saves the eimg ids as metatdata 
    input:
        eimgs_ids (np.array [int]) : event image ids
        eimgs_ts (np.array [int]) : time stamp of each event image
        targ_dir (str): directory to save to
    """
    metadata = {}

    for i, (id, t) in enumerate(zip(eimgs_ids, eimgs_ts)):
        metadata[str(i).zfill(6)] = {"warp_id":int(id),
                                     "appearance_id":int(id),
                                     "camera_id":0,
                                     "t":int(t)}
    
    with open(osp.join(targ_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def write_train_val_split(eimgs_ids, targ_dir):
    eimgs_ids = [str(int(e)).zfill(6) for e in eimgs_ids]
    save_path = osp.join(targ_dir, "dataset.json")

    train_ids = sorted(eimgs_ids)
    dataset_json = {
        "count":len(eimgs_ids),
        "num_exemplars":len(train_ids),
        "train_ids": eimgs_ids,
        "val_ids":[]
    }

    with open(save_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

def main():
    targ_dir = "adapt_carpet_formatted/ecam_set"
    src_dir = "generated_data/adapt_carpet"
    rgb_trigger_f = osp.join(src_dir, "sparse_intensity_imgs","triggers.txt")
    evs_f = osp.join(src_dir, "events.h5")
    intrinsics_f = osp.join(src_dir, "intrinsics.json")
    extrinsics_f = osp.join(src_dir, "traj.txt")
    max_t = 30*1e6
    
    
    time_delta =  5000
    intrxs = read_intrinsics_json(intrinsics_f)
    # intrxs = read_intrinsics(intrinsics_f)
    cx, cy = intrxs[0,2], intrxs[1,2]
    img_size = (2*cy, 2*cx)
    
    evs_buffer = EventBuffer(evs_f)
    cam_generator = CameraSpline(extrinsics_f)

    rgb_triggers = (np.round(np.loadtxt(rgb_trigger_f)/1e3)).astype(int)
    rgb_triggers = rgb_triggers[rgb_triggers <= max_t]
    eimgs, eimg_ts, eimgs_ids, trig_ids, eimg_end_ts = create_event_imgs(evs_buffer, 
                                                                         rgb_triggers, 
                                                                         time_delta=time_delta, 
                                                                         img_size = img_size,
                                                                         create_eimgs=False)

    if eimgs is not None:
        save_eimgs(eimgs, targ_dir)
        np.save("eimg_end_ts.npy", eimg_end_ts)
        del eimgs

    poses, orientations = cam_generator.interpolate(eimg_ts)
    extrinsic_targ_dir = osp.join(targ_dir, "camera")
    create_and_write_camera_extrinsics(extrinsic_targ_dir, orientations, poses, eimg_ts, intrxs)

    write_metadata(eimgs_ids, eimg_ts, targ_dir)

    # save the trig_ids; make the color camera ids the same
    np.save(osp.join(targ_dir, "trig_ids.npy"), trig_ids)

    # copy event to places
    shutil.copyfile(evs_f, osp.join(targ_dir, osp.basename(evs_f)))

    # create train valid split
    write_train_val_split(eimgs_ids, targ_dir)


if __name__ == "__main__":
    main()
