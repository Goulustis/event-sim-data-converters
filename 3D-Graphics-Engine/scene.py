from model import *
import glm

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
                add(Cube(app, pos=(x, y, -3), tex_id=tex_idx%2 * 2))
                # if tex_idx%2 == 0:
                #     add(Cube(app, pos=(x, y, -3), tex_id=tex_idx%2 * 2))
                # else:
                #     add(Cube(app, pos=(x, y, -3), tex_id="ori_carpet"))

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

        # translations
        self.dx = 0
        self.dy = -5
        self.dz = 0 #-20

        # rotations
        self.rx = 0
        self.ry = 0
        self.rz = 0

        # scale
        self.scale = 1


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
                add(Cube(app, pos=(x + self.dx, -s + self.dy, z + self.dz), 
                         rot=(self.rx, self.ry, self.rz), 
                         scale=(self.scale, self.scale, self.scale),
                         tex_id=tex_dic[tex_idx%2]))
                tex_idx += 1

        # wall
        for x in range(-n, n + 2, s):
            for y in range(0, n + 2, s):
                # add(Cube(app, pos=(x, y, -3), tex_id=tex_dic[tex_idx%2]))
                # add(Cube(app, pos=(x + self.dx, y + self.dy, -3 + self.dz), 
                #          rot=(self.rx, self.ry, self.rz), 
                #          scale=(self.scale, self.scale, self.scale),
                #          tex_id=tex_idx%2 * 2))
                add(Cube(app, pos=(x + self.dx, y + self.dy, -3 + self.dz), 
                         rot=(self.rx, self.ry, self.rz), 
                         scale=(self.scale, self.scale, self.scale),
                         tex_id="carpet"))

                tex_idx += 1

        # cat
        # sz = 0.2
        # add(Cat(app, pos=(self.dx - 2, 5 + self.dy, self.dz), 
        #              scale=(sz*self.scale, sz*self.scale, sz*self.scale)))
        sz1= 0.4
        z1 = -3 + self.scale + sz1
        add(Cube(app, pos=(self.dx - 0.5, 4.5 + self.dy, self.dz + z1),
                 rot=(0, 0, 20),
                 scale=(sz1, sz1, sz1),
                 tex_id="rocks"))
        
        sz2 = 0.3
        z2 =  -3 + self.scale + sz2
        add(Cube(app, pos=(self.dx+0.5, 5 + self.dy, self.dz + z2),
                 rot=(0, 0, 45),
                 scale=(sz2, sz2, sz2),
                 tex_id="grass"))
        

        sz3= 0.35
        z3 = -3 + self.scale + sz3
        add(Cube(app, pos=(self.dx + 1.7, 6 + self.dy, self.dz + z3),
                 rot=(0, 0, 0),
                 scale=(sz3, sz3, sz3),
                 tex_id="rocks"))

        sz3= 0.35
        z3 = -3 + self.scale + sz3
        add(Cube(app, pos=(self.dx + 1, 4 + self.dy, self.dz + z3),
                 rot=(0, 0, 0),
                 scale=(sz3, sz3, sz3),
                 tex_id="rocks"))
        

        sz4= 0.35
        z4 = -3 + self.scale + sz3
        add(Cube(app, pos=(self.dx - 1.5, 6.5 + self.dy, self.dz + z4),
                 rot=(0, 0, 0),
                 scale=(sz4, sz4, sz4),
                 tex_id="rocks"))

        # moving cube
        # self.moving_cube = MovingCube(app, pos=(0, 6, 8), scale=(3, 3, 3), tex_id=1)
        # add(self.moving_cube)
        # add(Cube(app, pos=(0,3,3), tex_id=1))
    def update(self):
        pass
        # self.moving_cube.rot.xyz = self.app.time



scene_cls_dict = {"robo" : Scene,
                  "carpet" : CarpetScene}