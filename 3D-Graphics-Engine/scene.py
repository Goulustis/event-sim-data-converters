from model import *

# NOTE: this is the cat scene
class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        # floor
        tex_dic = {1 : "white",
                   0 : "black"}
        tex_idx = 0
        n, s = 20, 2
        for x in range(-n, n + 2, s):
            for z in range(-n, n + 2, s):
                # add(Cube(app, pos=(x, -s, z)))
                add(Cube(app, pos=(x, -s, z), tex_id=tex_dic[tex_idx%2]))
                tex_idx += 1

        # wall
        for x in range(-n, n + 2, s):
            for y in range(0, n + 2, s):
                # add(Cube(app, pos=(x, y, -3), tex_id=tex_dic[tex_idx%2]))
                add(Cube(app, pos=(x, y, -3), tex_id=tex_idx%2 * 2))

                tex_idx += 1

        # cat
        sz = 0.2
        add(Cat(app, pos=(0, -1, 0), scale=(sz, sz, sz)))

        # moving cube
        # self.moving_cube = MovingCube(app, pos=(0, 6, 8), scale=(3, 3, 3), tex_id=1)
        # add(self.moving_cube)
        # add(Cube(app, pos=(0,3,3), tex_id=1))
    def update(self):
        pass
        # self.moving_cube.rot.xyz = self.app.time

class CarpetScene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        # floor
        tex_dic = {1 : "white",
                   0 : "black"}
        tex_idx = 0

        n, s = 20, 2.0
        for x in np.arange(-n, n + 2, s):
            for z in np.arange(-n, n + 2, s):
                # add(Cube(app, pos=(x, -s, z)))
                add(Cube(app, pos=(x, -s, z), scale=(s/2,s/2,s/2), tex_id="carpet"))
                tex_idx += 1

        # wall
        n, s = 20, 1.5
        tex_idx=1
        for x in np.arange(-n, n + 3, s):
            for y in np.arange(-2, n + 3, s):
                if x < 1:
                    add(Cube(app, pos=(x, y, -3), scale=(s/2,s/2,s/2), tex_id=tex_dic[tex_idx%2]))
                else:
                    add(Cube(app, pos=(x, y, -3), scale=(s/2,s/2,s/2), tex_id="carpet"))


                tex_idx += 1



    def update(self):
        pass
        # self.moving_cube.rot.xyz = self.app.time