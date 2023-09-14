import os
import os.path as osp
import pygame as pg
import moderngl as mgl
import sys
from model import *
from camera import Camera, PlayCamera
from camera_data.camera_spline import read_intrinsics, intrinsics_paths_dic
from light import Light
from mesh import Mesh
from scene import Scene, CarpetScene, scene_cls_dict
from scene_renderer import SceneRenderer
from OpenGL.GL import *
from PIL import Image
from tqdm import tqdm
import shutil
from engine_configs import SCENE, MODE

class GraphicsEngine:
    def __init__(self, win_size=(1600, 900), scene_cls=Scene):
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
        self.camera = Camera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = scene_cls(self)
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


class SimulatorEngine(GraphicsEngine):
    def __init__(self, win_size=(1600, 900), scene_cls = Scene, save_frame_dir = "dev_frames", save_mem=True):
        super().__init__(win_size, scene_cls)

        self.save_winsize = (win_size[0]//2, win_size[1]//2)
        self.camera = PlayCamera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = scene_cls(self)
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
        img = Image.fromarray(frame).convert('L')
        img = img.resize(self.save_winsize, resample=Image.LANCZOS)
        img.save(path)
    
    def save_all_imgs(self, targ_dir=None):
        if targ_dir is None:
            targ_dir = self.save_frame_dir
        
        for i, img in tqdm(enumerate(self.frames), total=len(self.frames), desc="saving frames"):
            save_path = osp.join(targ_dir, str(i).zfill(6) + ".png")
            self.save_img(img, save_path)
            


    def get_img(self):
        image_buffer = glReadPixels(0, 0, self.WIN_SIZE[0], self.WIN_SIZE[1], OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(self.WIN_SIZE[1], self.WIN_SIZE[0], 3)
        return image[::-1,::-1]



if __name__ == '__main__':
    fx, fy, cx, cy = read_intrinsics(intrinsics_paths_dic[f"{SCENE}_{MODE}".lower()])
    win_size = (int(cx*2), int(cy*2))

    if MODE == "run":
        app = GraphicsEngine(win_size=win_size, scene_cls=scene_cls_dict[SCENE])
    elif MODE == "render":
        app = SimulatorEngine(win_size=win_size, scene_cls=scene_cls_dict[SCENE], save_frame_dir="generated_imgs/cat_simple_2048")
    app.run()
