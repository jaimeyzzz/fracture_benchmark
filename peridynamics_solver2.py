import numpy as np
import taichi as ti

from neighbor_search2 import NeighborSearch2

MAX_NEIGHBOR_NUM = 32
MAX_BONDS_NUM = 16
EPS = 1e-6

FLUID = 0
BOUNDARY = 1

VALID = 0
BROKEN = 1

@ti.data_oriented
class PeridynamicsSolver2:
    # FIELDS
    mass = ti.field(ti.f32)
    radius = ti.field(ti.f32)
    label = ti.field(ti.i32)
    bondsNum = ti.field(ti.i32)
    bondsIdx = ti.field(ti.i32)
    bondsState = ti.field(ti.i32)
    neighbors = ti.field(ti.i32)
    neighborsNum = ti.field(ti.i32)
    oldNeighborsNum = ti.field(ti.i32)
    oldNeighbors = ti.field(ti.i32)
    # VECTORS
    position = ti.Vector.field(2, ti.f32)
    velocity = ti.Vector.field(2, ti.f32)
    velocityMid = ti.Vector.field(2, ti.f32)
    force = ti.Vector.field(2, ti.f32)
    bonds = ti.Vector.field(6, ti.f32)
    strings = ti.Vector.field(2, ti.f32)
    oldStrings = ti.Vector.field(2, ti.f32)
    # PARAMS
    gravity = ti.Vector.field(2, ti.f32)

    def __init__(self, numParticles, kn, kc, gammac, us, h):
        self.N = numParticles
        self.kn = kn
        self.kc = kc
        self.gammac = gammac
        self.us = us
        self.h = h
        ti.root.dense(ti.i, self.N).place(self.mass)
        ti.root.dense(ti.i, self.N).place(self.radius)
        ti.root.dense(ti.i, self.N).place(self.position)
        ti.root.dense(ti.i, self.N).place(self.velocity)
        ti.root.dense(ti.i, self.N).place(self.velocityMid)
        ti.root.dense(ti.i, self.N).place(self.force)
        ti.root.dense(ti.i, self.N).place(self.label)
        ti.root.dense(ti.i, self.N).place(self.neighborsNum)
        ti.root.dense(ti.i, self.N).place(self.oldNeighborsNum)
        ti.root.dense(ti.i, self.N).place(self.bondsNum)
        ti.root.dense(ti.i, self.N).dense(ti.j, MAX_NEIGHBOR_NUM).place(self.neighbors)
        ti.root.dense(ti.i, self.N).dense(ti.j, MAX_NEIGHBOR_NUM).place(self.oldNeighbors)
        ti.root.dense(ti.i, self.N).dense(ti.j, MAX_NEIGHBOR_NUM).place(self.strings)
        ti.root.dense(ti.i, self.N).dense(ti.j, MAX_NEIGHBOR_NUM).place(self.oldStrings)
        ti.root.dense(ti.i, self.N).dense(ti.j, MAX_BONDS_NUM).place(self.bonds)
        ti.root.dense(ti.i, self.N).dense(ti.j, MAX_BONDS_NUM).place(self.bondsIdx)
        ti.root.dense(ti.i, self.N).dense(ti.j, MAX_BONDS_NUM).place(self.bondsState)
        ti.root.dense(ti.i, 1).place(self.gravity)
    
    def init(self, npPos, npVel, npMass, npLabels, npRadius, ns, bs):
        self.position.from_numpy(npPos)
        self.velocity.from_numpy(npVel)
        self.velocityMid.from_numpy(npVel)
        self.mass.from_numpy(npMass)
        self.label.from_numpy(npLabels)
        self.radius.from_numpy(npRadius)
        self.oldNeighborsNum.fill(0)
        self.gravity.from_numpy(np.array([0.0,-9.81],dtype=np.float32).reshape((1,2)))
        self.ns = ns

        # init bonds
        self.bs = bs
        self.bs.updateCells(self.position)
        self.initBonds()
    
    @ti.kernel
    def initBonds(self):
        for i in self.position:
            li = self.label[i]
            if li == FLUID:
                pi = self.position[i]
                ri = self.radius[i]
                cell = self.bs.getCell(pi)
                cnt = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                    neighborCell = cell + offs
                    if self.bs.isCellInRange(neighborCell):
                        for k in range(self.bs.cellsNum[neighborCell]):
                            j = self.bs.cells[neighborCell, k]
                            pj = self.position[j]
                            rj = self.radius[j]
                            dx = (pi - pj).norm()
                            if i != j and dx < self.h:
                                self.bonds[i, cnt] = ti.Vector([dx, 0.0, 0.0, 0.0, 0.0, 0.0])
                                self.bondsState[i, cnt] = VALID
                                self.bondsIdx[i, cnt] = j
                                cnt += 1
                self.bondsNum[i] = cnt

    @ti.kernel
    def computeNeighbors(self):
        for i in self.position:
            li = self.label[i]
            if li == FLUID:
                pi = self.position[i]
                ri = self.radius[i]
                cell = self.ns.getCell(pi)
                cnt = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
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
    def computeCollision(self, dt:ti.f32):
        for i in self.force:
            li = self.label[i]
            if li == FLUID:
                mi = self.mass[i]
                xi = self.position[i]
                ri = self.radius[i]
                vi = self.velocity[i]
                f = ti.Vector([0.0, 0.0])
                for idx in range(self.neighborsNum[i]):
                    j = self.neighbors[i, idx]
                    bondExist = False
                    for iidx in range(self.bondsNum[i]):
                        k = self.bondsIdx[i, iidx]
                        if j == k and self.bondsState[i, iidx] == VALID:
                            bondExist = True
                    if bondExist: continue
                    xj = self.position[j]
                    rj = self.radius[j]
                    lj = self.label[j]
                    mj = self.mass[j]
                    vj = self.velocity[j]
                    dx = xi - xj
                    dxNorm = dx.norm()
                    overlap = ri + rj - dxNorm
                    n = dx / dxNorm
                    mij = mi * mj / (mi + mj) if lj == FLUID else mi
                    vij = vi - vj
                    # repulsion
                    kc = self.kc
                    gamma = 2.0 * self.gammac * ti.sqrt(mij * kc)
                    fn = kc * overlap - gamma * vij.dot(n)
                    f += fn * n
                    # friction
                    xiS = ti.Vector([0.0, 0.0])
                    for k in range(self.oldNeighborsNum[i]):
                        if j == self.oldNeighbors[i, k]:
                            xiS = self.oldStrings[i, k]
                            break
                    vt = vij - (n.dot(vij)) * n
                    xiS -= (n.dot(xiS)) * n
                    xiS += vt * dt
                    fs = -self.kc * xiS - gamma * vt
                    fsNorm = fs.norm()
                    if (fsNorm < EPS): continue
                    if (fsNorm >= fn * self.us):
                        fs = fs / fsNorm * fn * self.us
                        xiS = (fs + gamma * vt) / -kc
                    f += fs
                    self.strings[i, idx] = xiS
                self.force[i] += f
        # copy to old
        for i in self.position:
            self.oldNeighborsNum[i] = self.neighborsNum[i]
        for i in self.position:
            for idx in range(self.neighborsNum[i]):
                self.oldNeighbors[i, idx] = self.neighbors[i, idx]
                self.oldStrings[i, idx] = self.strings[i, idx]
    
    @ti.kernel
    def computeBond(self, dt: ti.f32):
        for i in self.force:
            li = self.label[i]
            if li == FLUID:
                mi = self.mass[i]
                xi = self.position[i]
                f = ti.Vector([0.0, 0.0])
                for idx in range(self.bondsNum[i]):
                    j = self.bondsIdx[i, idx]
                    state = self.bondsState[i, idx]
                    if state != VALID: continue
                    bond = self.bonds[i, idx]
                    l0 = bond[0]
                    xj = self.position[j]
                    dx = xi - xj
                    l = dx.norm()
                    n = dx / l
                    fn = -self.kn * (l / l0 - 1) * n
                    tao = fn.norm()
                    if tao > 1e4:
                        self.bondsState[i, idx] = BROKEN
                    f += fn                   
                self.force[i] += f
    
    @ti.kernel
    def computeGravity(self, dt: ti.f32):
        for i in self.force:
            li = self.label[i]
            if li == FLUID:
                mi = self.mass[i]
                self.force[i] += self.gravity[0] * mi
                
    @ti.kernel
    def updatePosition(self, dt: ti.f32):
        for i in self.position:
            self.force[i] = ti.Vector([0.0, 0.0])
            self.position[i] += self.velocityMid[i] * dt

    @ti.kernel
    def updateVelocity(self, dt: ti.f32):
        for i in self.velocity:
            li = self.label[i]
            if li == FLUID:
                mi = self.mass[i]
                acc = self.force[i] / mi
                self.velocity[i] = self.velocityMid[i] + 0.5 * acc * dt
                self.velocityMid[i] += acc * dt

    def updateNeighbors(self):
        self.neighborsNum.fill(0)
        self.ns.updateCells(self.position)
        self.computeNeighbors()

    def update(self, dt):
        self.updatePosition(dt)
        self.updateNeighbors()

        self.computeBond(dt)
        self.computeCollision(dt)
        self.computeGravity(dt)
        # self.computeBoundary(dt)

        self.updateVelocity(dt)