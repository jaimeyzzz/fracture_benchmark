import numpy as np
import taichi as ti

from neighbor_search2 import NeighborSearch2
from solver_base2 import SolverBase2

@ti.data_oriented
class SolverBdem2(SolverBase2):
    def __init__(self, scene, neighborSearch):
        super().__init__(scene, neighborSearch)
        self.kn = scene.kn
        self.kt = scene.kt
        self.kc = scene.kc
        self.gammac = scene.gammac
        self.us = scene.us
        self.h = scene.h
        self.sigmac = scene.sigma
        self.tauc = scene.tau

        self.momentOfInertia = ti.field(ti.f32)
        self.rotation = ti.field(ti.f32)
        self.angularVelocity = ti.field(ti.f32)
        self.angularVelocityMid = ti.field(ti.f32)
        self.torsion = ti.field(ti.f32)
        self.bondsLength = ti.field(ti.f32)
        self.bondsSigma = ti.field(ti.f32)
        self.bondsTao = ti.field(ti.f32)
        self.bondsDirection = ti.Vector.field(2, ti.f32)
        ti.root.dense(ti.i, self.N).place(self.momentOfInertia)
        ti.root.dense(ti.i, self.N).place(self.rotation)
        ti.root.dense(ti.i, self.N).place(self.angularVelocity)
        ti.root.dense(ti.i, self.N).place(self.angularVelocityMid)
        ti.root.dense(ti.i, self.N).place(self.torsion)
        ti.root.dense(ti.i, self.M).place(self.bondsLength)
        ti.root.dense(ti.i, self.M).place(self.bondsSigma)
        ti.root.dense(ti.i, self.M).place(self.bondsTao)
        ti.root.dense(ti.i, self.M).place(self.bondsDirection)
   
    def init(self):
        super().init()
        self.initProperties()
        self.initBonds()

    @ti.kernel
    def initProperties(self):
        for i in self.momentOfInertia:
            li = self.label[i]
            if li == self.scene.FLUID:
                self.momentOfInertia[i] = 0.5 * self.mass[i] * self.radius[i] ** 2
   
    @ti.kernel
    def initBonds(self):
        for i in self.position:
            li = self.label[i]
            if li == self.scene.FLUID:
                xi = self.position[i]
                for k in range(self.bondsAccum[i], self.bondsAccum[i + 1]):
                    j = self.bondsIdx[k]
                    xj = self.position[j]
                    dx = xj - xi
                    dxNorm = dx.norm()
                    self.bondsState[k] = self.scene.BOND_VALID
                    self.bondsLength[k] = dxNorm
                    self.bondsDirection[k] = dx / dxNorm
                    self.bondsSigma[k] = self.kn * self.sigmac
                    self.bondsTao[k] = self.kt * self.tauc

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
                    bondExist = False
                    for iidx in range(self.bondsAccum[i], self.bondsAccum[i + 1]):
                        k = self.bondsIdx[iidx]
                        if j == k and self.bondsState[iidx] == self.scene.BOND_VALID:
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
    
    @ti.kernel
    def computeBond(self, dt: ti.f32):
        for i in self.force:
            li = self.label[i]
            if li == self.scene.FLUID:
                mi = self.mass[i]
                Ii = self.momentOfInertia[i]
                pi = self.position[i]
                qi = self.rotation[i]
                vi = self.velocityMid[i]
                wi = self.angularVelocity[i]
                force = ti.Vector([0.0, 0.0])
                torsion = 0.0
                sigmaSum = 0.0
                tauSum = 0.0
                sigmaMax = 0.0
                tauMax = 0.0
                count = 0
                for idx in range(self.bondsAccum[i], self.bondsAccum[i + 1]):
                    j = self.bondsIdx[idx]
                    state = self.bondsState[idx]
                    if state != self.scene.BOND_VALID: continue
                    mj = self.mass[j]
                    Ij = self.momentOfInertia[j]
                    lj = self.label[j]
                    pj = self.position[j]
                    qj = self.rotation[j]
                    vj = self.velocity[j]
                    wj = self.angularVelocity[j]
                    # bond params
                    l0 = self.bondsLength[idx]
                    sigma0 = self.bondsSigma[idx]
                    tao0 = self.bondsTao[idx]
                    d = self.bondsDirection[idx]
                    # params
                    r0 = l0 / 2.0
                    s0 = 2.0 * r0
                    I0 = 2.0 / 3.0 * r0 * r0 * r0
                    l = (pj - pi).norm()
                    dl = l - l0
                    n = (pj - pi) / l
                    t = ti.Vector([-n[1],n[0]])
                    thetai = self.clampRad(ti.atan2(d[1],d[0])+qi-ti.atan2(n[1],n[0]))
                    thetaj = self.clampRad(ti.atan2(d[1],d[0])+qj-ti.atan2(n[1],n[0]))

                    kn = self.kn * s0 / l0
                    kt = self.kt * I0 / l0 / l
                    km = self.kt * I0 / l0
                    mij = mi * mj / (mi + mj) if lj == self.scene.FLUID else mi
                    Iij = Ii * Ij / (Ii + Ij) if lj == self.scene.FLUID else Ii
                    gamman = 2.0 * self.gammac * ti.sqrt(mij * kn)
                    gammat = 2.0 * self.gammac * ti.sqrt(mij * kt / l)
                    gammam = 2.0 * self.gammac * ti.sqrt(Iij * km)
                    # tensile
                    fn = kn * dl * n
                    vn = (vj - vi).dot(n) * n
                    fnDamp = gamman * vn
                    # shear
                    ft = -kt * (thetai + thetaj) * t
                    vt = (vj - vi) - vn
                    ftDamp = gammat * vt
                    # bend
                    mt = km * (thetaj - thetai)
                    mtDamp = gammam * (wj - wi)
                    # fracture
                    sigma = (fn + fnDamp).norm() / s0 + ti.abs(mt + mtDamp) * r0 / I0
                    tao = (ft + ftDamp).norm() / s0
                    sigmaSum += sigma / sigma0
                    tauSum += tao / tao0
                    sigmaMax = ti.max(sigmaMax, sigma / sigma0)
                    tauMax = ti.max(tauMax, tao / tao0)
                    count += 1
                    if (((dl > 0.0) and sigma > sigma0) or tao > tao0):
                        self.bondsState[idx] = self.scene.BOND_BROKEN
                        continue
                    force += fn + fnDamp
                    force += ft + ftDamp
                    torsion += mt + mtDamp
                    torsion += (ft + ftDamp).cross(-0.5 * l * n)
                self.force[i] += force
                self.torsion[i] += torsion

                # self.color[i] = ti.Vector([sigmaSum / count / self.sigmac / self.kn, 0.0, 0.0])
                self.color[i] = ti.Vector([sigmaMax, 0.0, 0.0])
            
    @ti.kernel
    def updatePosition(self, dt: ti.f32):
        for i in self.position:
            self.force[i] = ti.Vector([0.0, 0.0])
            self.torsion[i] = 0.0
            self.position[i] += self.velocityMid[i] * dt; 
            self.rotation[i] += self.angularVelocityMid[i] * dt; 

    @ti.kernel
    def updateVelocity(self, dt: ti.f32):
        for i in self.velocityMid:
            li = self.label[i]
            if li == self.scene.FLUID:
                mi = self.mass[i]
                Ii = self.momentOfInertia[i]
                acc = self.force[i] / mi
                self.velocity[i] = self.velocityMid[i] + 0.5 * acc * dt
                self.velocityMid[i] += acc * dt
                angularAcc = self.torsion[i] / Ii
                self.angularVelocity[i] = self.angularVelocityMid[i] + 0.5 * angularAcc * dt
                self.angularVelocityMid[i] += angularAcc * dt

    def update(self, dt):
        self.updateTime(dt)

        self.updatePosition(dt)
        self.updateNeighbors()

        self.computeBond(dt)
        self.load.fill(0)
        self.computeCollision(dt)
        self.computeGravity(dt)
        # self.computeBoundary(dt)

        self.updateVelocity(dt)

    @ti.func
    def clampRad(self, theta):
        while (theta <= -np.pi):
            theta += 2.0 * np.pi
        while (theta > np.pi):
            theta -= 2.0 * np.pi
        return theta
