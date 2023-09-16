import os
import os.path as osp
import pygame as pg
import moderngl as mgl
import sys
from model import *
from camera import Camera, PlayCamera
from camera_data.camera_spline import read_intrinsics, intrinsics_path
from light import Light
from mesh import Mesh
from scene import Scene
from scene_renderer import SceneRenderer
from OpenGL.GL import *
from PIL import Image
from tqdm import tqdm
import shutil
import argparse

import cv2

from config_cnsts import ALIAS_SCALE

USE_UINT8 = True
class GraphicsEngine:
    def __init__(self, win_size=(1600, 900)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(False)
        pg.mouse.set_visible(True)
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0
        # light
        self.light = Light(position=(-3,2,2))
        # camera
        # self.camera = Camera(self)
        self.camera = PlayCamera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = Scene(self)
        # renderer
        self.scene_renderer = SceneRenderer(self)

        # extra attributes
        self.attr = {}

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                self.scene_renderer.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # render scene
        self.scene_renderer.render()
        # swap buffers
        pg.display.flip()

    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    def run(self):
        while True:
            self.get_time()
            self.check_events()
            self.camera.update()
            self.render()
            self.delta_time = self.clock.tick(120)


class SimulatorEngine:
    def __init__(self, win_size=(1600, 900), save_frame_dir = "dev_frames", save_mem=True):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        self.save_winsize = (win_size[0]//ALIAS_SCALE, win_size[1]//ALIAS_SCALE)

        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        # pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF | pg.HIDDEN)
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(False)
        pg.mouse.set_visible(True)
        
        
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        self.time = 0
        self.delta_time = 0
        # light
        self.light = Light(position=(-3,2,2))
        # camera
        # self.camera = Camera(self)
        self.camera = PlayCamera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = Scene(self)
        # renderer
        self.scene_renderer = SceneRenderer(self)

        # extra attributes:
        self.attr = {"done":False}
        self.save_mem = save_mem
        self.frames = []
        self.save_frame_dir = save_frame_dir
        if osp.exists(self.save_frame_dir):
            shutil.rmtree(self.save_frame_dir)

        os.makedirs(self.save_frame_dir, exist_ok=True)

    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18))
        # render scene
        self.scene_renderer.render()
        # swap buffers
        pg.display.flip()


    def run(self):
        print("simulating")
        curr_idx = 0
        while not (self.attr["done"]):
            self.camera.update()
            self.render()
            img = self.get_img()

            if self.save_mem:
                path = osp.join(self.save_frame_dir, str(curr_idx).zfill(6) + ".png")
                self.save_img(img, path)
                curr_idx += 1
            else:
                self.frames.append(img)

        if not self.save_mem:
            self.save_all_imgs()
    
    def save_img(self, frame:np.ndarray, path:str):
        # if USE_UINT8:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = cv2.resize(frame, (self.save_winsize), interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.resize(frame, (self.save_winsize), interpolation=cv2.INTER_AREA)
        frame = frame[...,::-1]
        cv2.imwrite(path, frame)
        # mode = "I;16" if frame.dtype == np.uint16 else "RGB"
        # img = Image.fromarray(frame, mode) #.convert('L')
        # img = img.resize(self.save_winsize, resample=Image.LANCZOS)
        # img.save(path)
        
    
    def save_all_imgs(self, targ_dir=None):
        if targ_dir is None:
            targ_dir = self.save_frame_dir
        
        for i, img in tqdm(enumerate(self.frames), total=len(self.frames), desc="saving frames"):
            save_path = osp.join(targ_dir, str(i).zfill(6) + ".png")
            self.save_img(img, save_path)
            


    def get_img(self):
        if USE_UINT8:
            image_buffer = glReadPixels(0, 0, self.WIN_SIZE[0], self.WIN_SIZE[1], OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
            image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(self.WIN_SIZE[1], self.WIN_SIZE[0], 3)
        else:
            image_buffer = glReadPixels(0, 0, self.WIN_SIZE[0], self.WIN_SIZE[1], OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_SHORT)
            image = np.frombuffer(image_buffer, dtype=np.uint16).reshape(self.WIN_SIZE[1], self.WIN_SIZE[0], 3)
        return image[::-1,::-1]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="generated_imgs/dev")
    args = parser.parse_args()

    USE_UINT8 = False

    fx, fy, cx, cy = read_intrinsics(intrinsics_path)
    win_size = (int(cx*2), int(cy*2))
    # app = GraphicsEngine(win_size=win_size)
    # app = SimulatorEngine(win_size=win_size, save_frame_dir="generated_imgs/carpet_tex_2048")
    app = SimulatorEngine(win_size=win_size, save_frame_dir=args.outdir)
    app.run()

