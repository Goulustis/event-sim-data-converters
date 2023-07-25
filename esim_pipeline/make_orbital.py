import numpy as np
import glob
import os.path as osp
import os
import math
import json

from utils import read_json, make_camera
from tensorflow_graphics.geometry.representation.ray import triangulate as ray_triangulate
from nerfies.camera import Camera
from tqdm import tqdm

def triangulate_rays(origins, directions):
  origins = origins[np.newaxis, ...].astype('float32')
  directions = directions[np.newaxis, ...].astype('float32')
  weights = np.ones(origins.shape[:2], dtype=np.float32)
  points = np.array(ray_triangulate(origins, origins + directions, weights))
  return points.squeeze()

def read_all_cams(cam_dir):
    cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))

    cams = []
    for cam_f in tqdm(cam_fs, desc="loading cameras"):
        cams.append(Camera.from_json(cam_f))
    
    return cams

def points_bound(points):
  """Computes the min and max dims of the points."""
  min_dim = np.min(points, axis=0)
  max_dim = np.max(points, axis=0)
  return np.stack((min_dim, max_dim), axis=1)

def points_bounding_size(points):
  """Computes the bounding size of the points from the bounding box."""
  bounds = points_bound(points)
  return np.linalg.norm(bounds[:, 1] - bounds[:, 0])


def save_orbit_cams(scene_dir, cameras):
    orbit_dir = osp.join(scene_dir, "camera-paths", "orbit-mild")
    os.makedirs(orbit_dir, exist_ok=True)

    for i, camera in enumerate(cameras):
        camera_path = osp.join(orbit_dir, f'{i:06d}.json')
        print(f'Saving camera to {camera_path!s}')
        with open(camera_path, "w") as f:
            json.dump(camera.to_json(), f, indent=2)




def main():
    scene_dir = "/ubc/cs/research/kmyi/matthew/projects/DyNeRF/datasets/ShakeCarpet1_formatted/colcam_set"
    cam_dir = osp.join(scene_dir, "camera")

    ref_cameras = read_all_cams(cam_dir)
    origins = np.array([c.position for c in ref_cameras])
    directions = np.array([c.optical_axis for c in ref_cameras])
    # look_at = triangulate_rays(origins, directions)
    ref_n = len(directions)//2
    look_at = origins[ref_n] + 30*directions[ref_n]
    print('look_at', look_at)

    avg_position = np.mean(origins, axis=0)
    print('avg_position', avg_position)

    up = -np.mean([c.orientation[..., 1] for c in ref_cameras], axis=0)
    print('up', up)

    bounding_size = points_bounding_size(origins) / 2
    x_scale =   0.75# @param {type: 'number'}
    y_scale = 0.75  # @param {type: 'number'}
    xs = x_scale * bounding_size
    ys = y_scale * bounding_size
    radius = 0.75  # @param {type: 'number'}
    num_frames = 100  # @param {type: 'number'}

    ref_camera = ref_cameras[len(ref_cameras)//2]
    print(ref_camera.position)
    z_offset = -0.1

    angles = np.linspace(0, 2*math.pi, num=num_frames)
    positions = []
    for angle in angles:
        x = np.cos(angle) * radius * xs
        y = np.sin(angle) * radius * ys

        position = np.array([x, y, z_offset])
        # Make distance to reference point constant.
        position = avg_position + position
        positions.append(position)

    positions = np.stack(positions)

    orbit_cameras = []
    for position in positions:
        camera = ref_camera.look_at(position, look_at, up)
        orbit_cameras.append(camera)
    
    save_orbit_cams(scene_dir, orbit_cameras)


if __name__ == "__main__":
    main()