from utils import gen_colcam_triggers, read_intrinsics, make_camera, gen_cams, EventBuffer
from synthetic_ev_scene.camera_spline import CameraSpline
from eimg_maker import create_event_imgs
import os
import os.path as osp
import numpy as np
import json
import shutil
import argparse

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
        camera = make_camera(R, pose, intr_mtx)
        cameras.append(camera)
        targ_cam_path = osp.join(extrinsic_dir, str(i).zfill(6) + ".json")
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

def write_train_valid_split(eimgs_ids, targ_dir):
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

def main(dst_dir,rgb_dir, evs_f, intrinsics_f, camspline_f, cam_mode="smooth"):
    #### inputs ###
    # dst_dir = "synth_robo_col_evs"
    # src_dir = "synthetic_ev_scene"
    # cam_mode = "smooth"  # one of ['smooth', 'lerp']
    time_delta =  5000


    ###############
    # targ_dir = osp.join(targ_home, "/ecam_set")
    # rgb_dir = "synthetic_ev_scene/coarse_frames/gamma"
    # evs_f = "synthetic_ev_scene/events.hdf5"
    # intrinsics_f = "synthetic_ev_scene/intrinsics.json"
    # targ_dir = osp.join(dst_dir, "ecam_set")
    # rgb_dir = osp.join(src_dir, "clear_coarse_frames/linear")
    # evs_f = osp.join(src_dir,"rgb_events.hdf5")
    # # evs_f = osp.join(src_dir,"robo_tex_gamma_events_4x.hdf5")
    # intrinsics_f = osp.join(src_dir, "intrinsics.json")
    # camspline_f = osp.join(src_dir, "camera_spline.npy")

    targ_dir = dst_dir
    
    intrxs = read_intrinsics(intrinsics_f)
    cx, cy = intrxs[0,2], intrxs[1,2]
    img_size = (2*cy, 2*cx)
    
    evs_buffer = EventBuffer(evs_f)
    cam_generator = CameraSpline(camspline_f, mode = cam_mode)

    rgb_triggers = gen_colcam_triggers(rgb_dir)
    eimgs, eimg_ts, eimgs_ids, trig_ids, eimg_end_ts = create_event_imgs(evs_buffer, rgb_triggers, 
                                                                        time_delta=time_delta, 
                                                                        img_size = img_size,
                                                                        create_eimgs=True)

    save_eimgs(eimgs, targ_dir)
    np.save("eimg_end_ts.npy", eimg_end_ts)
    del eimgs

    poses, orientations = gen_cams(eimg_ts, cam_generator)
    extrinsic_targ_dir = osp.join(targ_dir, "camera")
    create_and_write_camera_extrinsics(extrinsic_targ_dir, orientations, poses, eimg_ts, intrxs)

    write_metadata(eimgs_ids, eimg_ts, targ_dir)

    # save the trig_ids; make the color camera ids the same
    np.save(osp.join(targ_dir, "trig_ids.npy"), trig_ids)

    # copy event to places
    # shutil.copyfile(evs_f, osp.join(targ_dir, osp.basename(evs_f)))

    # create train valid split
    write_train_valid_split(eimgs_ids, targ_dir)


if __name__ == "__main__":
    ## assuming intrinsics to be the same as original robot scene
    data_dir = osp.join(osp.dirname(osp.realpath(__file__)), "synthetic_ev_scene")
    default_intrinsics_f = osp.join(data_dir, "intrinsics.json")
    default_camspline_f = osp.join(data_dir, "camera_spline.npy")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dst_dir", help="home directory of the dataset to be saved; save dir (eg .../scene/ecam_set)", default="")
    parser.add_argument("--coarse_rgb_dir", help="directory containing color camera dataset images, use to determine gap to accumulate", default="")
    parser.add_argument("--evs_f", help="event file to accumulate by", default="")
    parser.add_argument("--intrinsics_f", help="intrinsics of the generated scene", default=default_intrinsics_f)
    parser.add_argument("--camspline_f", help="synthetic robot camera spline format .npy", default=default_camspline_f)
    parser.add_argument("--cam_mode", choices=["smooth", "lerp"],default="smooth")
    args = parser.parse_args()

    main(args.dst_dir, args.coarse_rgb_dir, args.evs_f, args.intrinsics_f, args.camspline_f, args.cam_mode)