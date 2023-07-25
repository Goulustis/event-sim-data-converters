import glob
from utils import read_intrinsics, make_camera, read_poses_bounds, read_intrinsics_json, extract_image_ids
from camera_spline import CameraSpline
import numpy as np
import os
import os.path as osp
import json
import shutil


def create_dataset_dict(img_ids):
    val_ids = img_ids[::20]
    train_ids = sorted(set(img_ids) - set(val_ids))
    return {
        'count':len(img_ids),
        'num_exemplars': len(train_ids),
        'ids':img_ids,
        'train_ids':train_ids,
        "val_ids": val_ids
    }

def create_metadata_dict(train_ids, val_ids, img_trig_t_dic, img_trig_id_dic):
    metadata_dict = {}

    for i, img_id in enumerate(train_ids):
        metadata_dict[img_id] = {
            'warp_id':img_trig_id_dic[img_id],
            'appearance_id': img_trig_id_dic[img_id],
            'camera_id':0,
            't':img_trig_t_dic[img_id]
        }
    
    for i, img_id in enumerate(val_ids):
        metadata_dict[img_id] = {
            'warp_id':img_trig_id_dic[img_id],
            'appearance_id': img_trig_id_dic[img_id],
            'camera_id':0,
            't':img_trig_t_dic[img_id]
        }
    
    return metadata_dict

def save_all_cams(img_ids, orientations, poses, trig_ts, targ_dir, intrx_mtx):
    cam_dir = osp.join(targ_dir, "camera")
    os.makedirs(cam_dir, exist_ok=True)

    for img_id, ori, pos, trig_t in zip(img_ids, orientations, poses, trig_ts):
        cam_path = osp.join(cam_dir, f'{img_id}.json')
        camera = make_camera(ori, pos, intrx_mtx)
        cam_json = camera.to_json()
        cam_json['t'] = str(trig_t)
        with open(cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)


def gen_points(w2cs, ts, intrinsics_f):
    _, bds = read_poses_bounds(intrinsics_f)
    Rs = w2cs.transpose(0,2,1)   # c2ws
    depth_vec = np.array([0, 0, 1])[None]

    depth_dirs = np.concatenate([depth_vec@R for R in Rs]) * bds.max()
    # depth_dirs = np.concatenate([(R@depth_vec.T).T for R in Rs]) * bds.max()
    return ts + depth_dirs

def get_bbox_corners(points):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return np.stack([lower, upper])

def viz_params(pnts, orientations, poses, bbox_corners, near, far, intrxs, scene_center):
    print(f"near: {near}, far:{far}")
    import plotly.graph_objs as go
    def scatter_points(points, size=2):
        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=size),
        )
    cam_idx = 0
    camera = make_camera(orientations[cam_idx], poses[cam_idx], intrxs)
    near_points = camera.pixels_to_points(
        camera.get_pixel_centers()[::8, ::8], np.array(near)).reshape((-1, 3))
    far_points = camera.pixels_to_points(
        camera.get_pixel_centers()[::8, ::8], np.array(far)).reshape((-1, 3))
    
    data = [
        scatter_points(pnts),
        scatter_points(poses),
        scatter_points(bbox_corners),
        scatter_points(near_points),
        scatter_points(far_points),
        scatter_points(scene_center[None])
        ]

    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode='orbit')
    fig.show()

def main():
    targ_dir = "unif_carpet_2048/colcam_set"
    src_dir = "generated_data/unif_carpet_2048"

    img_dir = osp.join(src_dir, "sparse_intensity_imgs")
    rgb_trigger_f = osp.join(img_dir, "triggers.txt")
    extrinsics_f = osp.join(src_dir, "traj.txt")
    trig_ids_path = osp.join(osp.dirname(targ_dir),"ecam_set", "trig_ids.npy")
    intrinsics_f = osp.join(src_dir, "intrinsics.json")

    print("loading data")
    cam_generator = CameraSpline(extrinsics_f)
    trig_ts = (np.round(np.loadtxt(rgb_trigger_f)/1e3)).astype(int)
    
    trig_ids = np.load(trig_ids_path)

    # pnts = gen_points(cam_generator.w2cs, cam_generator.coords, intrinsics_f)
    img_ids = extract_image_ids(img_dir)
    # img_idxs = [e.split("/")[-1].split(".")[0] for e in sorted(glob.glob(osp.join(img_dir, "*.png")))]

    # intrx = read_intrinsics(intrinsics_f)
    intrx = read_intrinsics_json(intrinsics_f)

    cam_poses, orientations = cam_generator.interpolate(trig_ts)
    # all_pnts = np.concatenate([pnts, cam_poses])
    all_pnts = cam_poses #np.concatenate([pnts, cam_poses])

    os.makedirs(targ_dir, exist_ok=True)

    img_ids, trig_ts = img_ids[:len(trig_ids)], trig_ts[:len(trig_ids)]
    img_trig_t_dic = {}
    img_trig_id_dic = {}
    img_int_id_dic = {}
    for i, (img_id, trig_t, trig_id) in enumerate(zip(img_ids, trig_ts, trig_ids)):
        img_trig_t_dic[img_id] = int(trig_t)
        img_trig_id_dic[img_id] = int(trig_id)
        img_int_id_dic[img_id] = i
    
    # poses_c2w, bds = read_poses_bounds(intrinsics_f)
    # near, far = bds.min()/0.8, bds.max()*1.2
    near, far = 0.1, 40

    bbox_corners = get_bbox_corners(all_pnts)
    scene_center = bbox_corners.mean(axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))


    dataset_dic = create_dataset_dict(img_ids)
    metadata_dic = create_metadata_dict(dataset_dic["train_ids"], dataset_dic["val_ids"], 
                                        img_trig_t_dic, img_trig_id_dic)
    
    print('saving things')
    save_all_cams(img_ids, orientations, cam_poses, trig_ts, targ_dir, intrx)

    scene_path = osp.join(targ_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump({
      'scale': scene_scale,
      'center': scene_center.tolist(),
      'bbox': bbox_corners.tolist(),
      'near': near* scene_scale,
      'far': far * scene_scale,
    }, f, indent=2)
    
    metadata_path = osp.join(targ_dir, "metadata.json")
    dataset_path = osp.join(targ_dir, "dataset.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata_dic, f, indent=2)
    
    with open(dataset_path, "w") as f:
        json.dump(dataset_dic, f, indent=2)
    
    targ_img_dir = osp.join(targ_dir, "rgb", "1x")
    if not osp.exists(targ_img_dir):
        shutil.copytree(img_dir, targ_img_dir)

    # for sanity check
    # viz_params(pnts, orientations, cam_poses, bbox_corners, near, far, intrx)
    # viz_params(pnts, orientations, cam_poses, bbox_corners, near, far, intrx, scene_center)


if __name__ == "__main__":
    main()