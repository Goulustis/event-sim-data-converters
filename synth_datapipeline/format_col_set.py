import numpy as np
import os.path as osp
import shutil
import glob
from synthetic_ev_scene.camera_spline import CameraSpline
from utils import make_camera, gen_cams, gen_colcam_triggers, read_point_clouds, read_intrinsics, read_triggers
import pandas as pd
import nerfies.camera as cam
import os
import json
from tqdm import tqdm

def extract_image_ids(img_dir):
    img_fs = sorted(glob.glob(osp.join(img_dir, "*.png")))
    img_fs = [osp.basename(img_f).split(".")[0] for img_f in img_fs]
    return sorted(img_fs)


def filter_outlier_points(points, inner_percentile=0.95):
  """Filters outlier points."""
  outer = 1.0 - inner_percentile
  lower = outer / 2.0
  upper = 1.0 - lower
  centers_min = np.quantile(points, lower, axis=0)
  centers_max = np.quantile(points, upper, axis=0)
  result = points.copy()

  too_near = np.any(result < centers_min[None, :], axis=1)
  too_far = np.any(result > centers_max[None, :], axis=1)

  return result[~(too_near | too_far)]


def calc_cam_depth(pnts, cam_orientation, cam_pos):
    translated_points = pnts - cam_pos
    local_points = (np.matmul(cam_orientation, translated_points.T)).T
    return local_points[..., 2]


def estimate_near_far_single(pnts, cam_orientation, cam_pos, intrx):
    
    camera = make_camera(cam_orientation, cam_pos, intrx)
    pixels = camera.project(pnts)
    
    # depths = calc_cam_depth(pnts, cam_orientation, cam_pos)
    # depths = depths[depths > 0]
    depths = camera.points_to_local_points(pnts)[..., 2]
    in_frustum = (
      (pixels[..., 0] >= 0.0)
      & (pixels[..., 0] <= camera.image_size_x)
      & (pixels[..., 1] >= 0.0)
      & (pixels[..., 1] <= camera.image_size_y))
    depths = depths[in_frustum]

    # in_front_of_camera = depths > 0
    # depths = depths[in_front_of_camera]
    if len(depths) == 0:
        return None, None

    near = np.quantile(depths, 0.001)
    far = np.quantile(depths, 0.999)

    return near, far



def estimate_near_far(orientations, cam_poses, pnts: np.ndarray, intrx):
    results = []

    for ori, pos in tqdm(zip(orientations, cam_poses), total=len(cam_poses), desc="estimating near, far"):
        near, far = estimate_near_far_single(pnts, ori, pos, intrx)
        if near is not None:
            results.append({"near":near, "far":far})
    
    df = pd.DataFrame.from_records(results)
    near, far = df["near"].quantile(0.001)/0.8, df["far"].quantile(0.999)*1.2
    return near, far

def get_bbox_corners(points):
  lower = points.min(axis=0)
  upper = points.max(axis=0)
  return np.stack([lower, upper])


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

    for img_id, ori, pos, trig_t in tqdm(zip(img_ids, orientations, poses, trig_ts), desc="saving cams", total=len(img_ids)):
        cam_path = osp.join(cam_dir, f'{img_id}.json')
        camera = make_camera(ori, pos, intrx_mtx)
        cam_json = camera.to_json()
        # cam_json['t'] = trig_t
        cam_json['t'] = int(trig_t)
        cam_json["img_id"] = img_id
        with open(cam_path, "w") as f:
            json.dump(cam_json, f, indent=2)
    

def viz_params(pnts, orientations, poses, bbox_corners, near, far, intrxs):
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
    cam_idx = 100
    camera = make_camera(orientations[100], poses[100], intrxs)
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
        ]

    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode='orbit')
    fig.show()

def main():
    ######################### inputs here
    # targ_home = "cat_plain_formatted"
    # src_home = "cat_plain"
    targ_home = "cat_lerp_formatted"
    src_home = "cat_lerp"
    cam_mode = "lerp" # one of ['smooth', 'lerp']

    ############################
    img_dir = f"{src_home}/clear_coarse_frames/linear"
    targ_dir = f"{targ_home}/colcam_set"
    # targ_dir = "dev_colcam_sub"
    cam_gen_path = f"{src_home}/camera_spline.npy"
    trig_ids_path = f"{targ_home}/ecam_set/trig_ids.npy"
    
    # point cloud generated from: https://products.aspose.com/3d/net/point-cloud/glb/
    pnt_cloud_path = "synthetic_ev_scene/mecha_cropped.xyz"
    intrx_path = f"{src_home}/intrinsics.json"
    max_t = int(10*1e6)

    print("loading data")
    cam_generator = CameraSpline(cam_gen_path, mode = cam_mode)

    trig_f = osp.join(img_dir, "triggers.txt")
    if osp.exists(trig_f):
        trig_ts = read_triggers(trig_f)
    else:
        trig_ts = gen_colcam_triggers(img_dir, max_t)
    trig_ids = np.load(trig_ids_path)
    pnts = read_point_clouds(pnt_cloud_path)
    img_ids = extract_image_ids(img_dir)
    intrx = read_intrinsics(intrx_path)


    cam_poses, orientations = cam_generator.interpolate(trig_ts)
    all_pnts = np.concatenate([pnts, cam_poses])

    os.makedirs(targ_dir, exist_ok=True)

    img_ids, trig_ts = img_ids[:len(trig_ids)], trig_ts[:len(trig_ids)]
    img_trig_t_dic = {}
    img_trig_id_dic = {}
    img_int_id_dic = {}
    for i, (img_id, trig_t, trig_id) in enumerate(zip(img_ids, trig_ts, trig_ids)):
        img_trig_t_dic[img_id] = int(trig_t)
        img_trig_id_dic[img_id] = int(trig_id)
        img_int_id_dic[img_id] = i
    
    
    print("estimating scene params")
    near, far = estimate_near_far(orientations, cam_poses, all_pnts, intrx)
    bbox_corners = get_bbox_corners(all_pnts)
    scene_center = bbox_corners.mean(axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))
    
    dataset_dic = create_dataset_dict(img_ids)
    metadata_dic = create_metadata_dict(dataset_dic["train_ids"], dataset_dic["val_ids"], 
                                        img_trig_t_dic, img_trig_id_dic)
    
    print('saving things')
    save_all_cams(img_ids, orientations, cam_poses, trig_ts, targ_dir, intrx)

    # write scene json
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
    viz_params(pnts, orientations, cam_poses, bbox_corners, near, far, intrx)


if __name__ == "__main__":
    main()


