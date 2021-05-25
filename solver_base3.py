import numpy as np
import taichi as ti

from neighbor_search3 import NeighborSearch3

@ti.data_oriented
class SolverBase3:
    EPS = 1e-12
    def __init__(self, scene, neighborSearch):
        self.scene = scene
        self.ns = neighborSearch
        self.N = scene.N

        data = np.load(scene.npzFile)
        self.inclusive = data['inclusive']
        self.indices = data['indices']
        self.M = self.indices.shape[0]

        # FIELDS
        self.mass = ti.field(ti.f32)
        self.radius = ti.field(ti.f32)
        self.label = ti.field(ti.i32)
        self.bondsAccum = ti.field(ti.i32)
        self.bondsIdx = ti.field(ti.i32)
        self.bondsState = ti.field(ti.i32)
        self.neighbors = ti.field(ti.i32)
        self.neighborsNum = ti.field(ti.i32)
        self.oldNeighborsNum = ti.field(ti.i32)
        self.oldNeighbors = ti.field(ti.i32)
        # VECTORS
        self.position = ti.Vector.field(3, ti.f32)
        self.velocity = ti.Vector.field(3, ti.f32)
        self.velocityMid = ti.Vector.field(3, ti.f32)
        self.force = ti.Vector.field(3, ti.f32)
        self.bonds = ti.Vector.field(6, ti.f32)
        self.strings = ti.Vector.field(3, ti.f32)
        self.oldStrings = ti.Vector.field(3, ti.f32)
        self.color = ti.Vector.field(3, ti.f32)
        # PARAMS
        self.t = ti.field(ti.f32)
        self.gravity = ti.Vector.field(3, ti.f32)
        self.load = ti.Vector.field(3, ti.f32)
        ti.root.dense(ti.i, self.N).place(self.mass)
        ti.root.dense(ti.i, self.N).place(self.radius)
        ti.root.dense(ti.i, self.N).place(self.position)
        ti.root.dense(ti.i, self.N).place(self.velocity)
        ti.root.dense(ti.i, self.N).place(self.velocityMid)
        ti.root.dense(ti.i, self.N).place(self.force)
        ti.root.dense(ti.i, self.N).place(self.label)
        ti.root.dense(ti.i, self.N).place(self.neighborsNum)
        ti.root.dense(ti.i, self.N).place(self.oldNeighborsNum)
        ti.root.dense(ti.i, self.N).dense(ti.j, scene.MAX_NEIGHBOR_NUM).place(self.neighbors)
        ti.root.dense(ti.i, self.N).dense(ti.j, scene.MAX_NEIGHBOR_NUM).place(self.oldNeighbors)
        ti.root.dense(ti.i, self.N).dense(ti.j, scene.MAX_NEIGHBOR_NUM).place(self.strings)
        ti.root.dense(ti.i, self.N).dense(ti.j, scene.MAX_NEIGHBOR_NUM).place(self.oldStrings)
        ti.root.dense(ti.i, self.N).place(self.color)

        ti.root.dense(ti.i, self.N + 1).place(self.bondsAccum)
        ti.root.dense(ti.i, self.M).place(self.bonds)
        ti.root.dense(ti.i, self.M).place(self.bondsIdx)
        ti.root.dense(ti.i, self.M).place(self.bondsState)
        ti.root.dense(ti.i, 1).place(self.gravity)
        ti.root.dense(ti.i, 1).place(self.load)
        ti.root.dense(ti.i, 1).place(self.t)
    
    def init(self):
        self.position.from_numpy(self.scene.position)
        self.velocity.from_numpy(self.scene.velocity)
        self.velocityMid.from_numpy(self.scene.velocity)
        self.mass.from_numpy(self.scene.mass)
        self.label.from_numpy(self.scene.label)
        self.radius.from_numpy(self.scene.radius)
        self.oldNeighborsNum.fill(0)
        self.gravity.from_numpy(self.scene.gravity.reshape((-1,3)))
        self.bondsAccum.from_numpy(self.inclusive)
        self.bondsIdx.from_numpy(self.indices)

    @ti.kernel
    def computeNeighbors(self):
        for i in self.position:
            li = self.label[i]
            if li == self.scene.FLUID:
                pi = self.position[i]
                ri = self.radius[i]
                cell = self.ns.getCell(pi)
                cnt = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                    neighborCell = cell + offs
                    if self.ns.isCellInRange(neighborCell):
                        for k in range(self.ns.cellsNum[neighborCell]):
                            j = self.ns.cells[neighborCell, k]
                            rj = self.radius[j]
                            pj = self.position[j]
                            if i != j and (pi - pj).norm() < ri + rj:
                                self.neighbors[i, cnt] = j
                                cnt += 1
                self.neighborsNum[i] = cnt
    
    @ti.kernel
    def computeGravity(self, dt: ti.f32):
        for i in self.force:
            li = self.label[i]
            if li == self.scene.FLUID:
                mi = self.mass[i]
                self.force[i] += self.gravity[0] * mi
                
    @ti.kernel
    def updatePosition(self, dt: ti.f32):
        for i in self.position:
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.position[i] += self.velocityMid[i] * dt

    @ti.kernel
    def updateVelocity(self, dt: ti.f32):
        for i in self.velocity:
            li = self.label[i]
            if li == self.scene.FLUID:
                mi = self.mass[i]
                acc = self.force[i] / mi
                self.velocity[i] = self.velocityMid[i] + 0.5 * acc * dt
                self.velocityMid[i] += acc * dt

    def updateNeighbors(self):
        self.neighborsNum.fill(0)
        self.ns.updateCells(self.position)
        self.computeNeighbors()

    @ti.kernel
    def updateTime(self, dt: ti.f32):
        self.t[0] += dt

    def update(self, dt):
        raise NotImplementedError