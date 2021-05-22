import numpy as np
import taichi as ti

from neighbor_search2 import NeighborSearch2
from solver_base2 import SolverBase2

@ti.data_oriented
class SolverDem2(SolverBase2):
    def __init__(self, scene, neighborSearch):
        super().__init__(scene, neighborSearch)

        self.kn = scene.kn
        self.kc = scene.kc
        self.gammac = scene.gammac
        self.us = scene.us
        self.h = scene.h

    def init(self):
        super().init()
   
    @ti.kernel
    def computeCollision(self, dt:ti.f32):
        for i in self.force:
            li = self.label[i]
            if li == self.scene.FLUID:
                mi = self.mass[i]
                xi = self.position[i]
                ri = self.radius[i]
                vi = self.velocity[i]
                f = ti.Vector([0.0, 0.0])
                for idx in range(self.neighborsNum[i]):
                    j = self.neighbors[i, idx]
                    xj = self.position[j]
                    rj = self.radius[j]
                    lj = self.label[j]
                    mj = self.mass[j]
                    vj = self.velocity[j]
                    dx = xi - xj
                    dxNorm = dx.norm()
                    overlap = ri + rj - dxNorm
                    n = dx / dxNorm
                    mij = mi * mj / (mi + mj) if lj == self.scene.FLUID else mi
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
                    if (fsNorm < self.EPS): continue
                    if (fsNorm >= fn * self.us):
                        fs = fs / fsNorm * fn * self.us
                        xiS = (fs + gamma * vt) / -kc
                    f += fs
                    self.strings[i, idx] = xiS
                    if self.label[j] == self.scene.LOAD:
                        self.load[0] += f
                self.force[i] += f
        # copy to old
        for i in self.position:
            self.oldNeighborsNum[i] = self.neighborsNum[i]
        for i in self.position:
            for idx in range(self.neighborsNum[i]):
                self.oldNeighbors[i, idx] = self.neighbors[i, idx]
                self.oldStrings[i, idx] = self.strings[i, idx]
    


    
    def update(self, dt):
        self.updateTime(dt)
        self.updatePosition(dt)
        self.updateNeighbors()

        self.computeCollision(dt)
        self.computeGravity(dt)
        # self.computeBoundary(dt)

        self.updateVelocity(dt)

        self.computeColor()