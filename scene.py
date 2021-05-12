import json
import numpy as np
import os
import yaml

from plyfile import PlyData, PlyElement

class Scene:
    FLUID = 0
    BOUNDARY = 1
    LOAD = 2

    BOND_VALID = 0
    BOND_BROKEN = 1

    MAX_BONDS_NUM = 128
    MAX_NEIGHBOR_NUM = 32
    def __init__(self, path, name):
        self.mass = np.array([], dtype=float)
        self.radius = np.array([], dtype=float)
        self.label = np.array([], dtype=int)
        self.position = np.array([], dtype=float)
        self.velocity = np.array([], dtype=float)
        self.color = np.array([], dtype=float)

        self.lowerBound = np.float32([-0.5, -0.5])
        self.upperBound = np.float32([0.5, 0.5])
        self.gravity = np.float32([0.0, -9.81])
        # self.gravity = np.float32([0.0, 0.0])

        self.npzFile = os.path.join(path, name, '{}.npz'.format(name))
        self.importYaml(os.path.join(path, name, '{}.yaml'.format(name)))
        self.importPly(os.path.join(path, name, '{}.ply'.format(name)))

        self.rMin = np.min(self.radius)
        self.rMax = np.max(self.radius)
        self.mMin = np.min(self.mass[self.label == self.FLUID])
        self.mMax = np.max(self.mass[self.label == self.FLUID])
        self.r = self.rMin
        self.h = self.rMax * self.range

    def importYaml(self, yamlFilePath):
        with open(yamlFilePath, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.cfl = data['cfl']
        self.fps = data['fps']
        self.gammac = data['dampingRatio']
        self.rho = data['density']
        # bond
        self.range = data['bondRange']
        self.kn = data['bondNormalStiffness']
        self.kt = data['bondTangentialStiffness']
        self.sigma = data['bondNormalStrength']
        self.tao = data['bondTangentialStrength']
        # collision
        self.kc = data['collisionStiffness']
        self.us = data['frictionCoefficient']

    @staticmethod
    def plyLoadAttributes(vertexData, name, k):
        attrib = np.array([])
        for i in range(k):
            nameK = '{}{}'.format(name, i+1)
            attrib = np.append(attrib, vertexData.data[nameK]) 
        return attrib.reshape((-1,3), order='F')

    def importPly(self, plyFilePath):
        with open(plyFilePath, 'rb') as f:
            plyData = PlyData.read(f)
        # IMPORT PLY ARRAY
        vertexData = plyData['vertex']
        self.N = vertexData.data.shape[0]
        # scalar data
        self.mass = np.array(vertexData.data['mass'])
        self.radius = np.array(vertexData.data['pscale'])
        self.label = np.array(vertexData.data['label'])
        # vector data
        x = np.array(vertexData.data['x'])
        y = np.array(vertexData.data['y'])
        z = np.array(vertexData.data['z'])
        self.position = np.transpose(np.array([x, y, z]), axes=[1, 0])
        r = np.array(vertexData.data['red'])
        g = np.array(vertexData.data['green'])
        b = np.array(vertexData.data['blue'])
        self.color = np.transpose(np.array([r, g, b]), axes=[1, 0])
        self.velocity = self.plyLoadAttributes(vertexData, 'velocity', 3)

scene = Scene('scene2', 'bending')