import numpy as np
import taichi as ti

from neighbor_search3 import NeighborSearch3
from solver_base3 import SolverBase3

###### static methods #####
@ti.func
def compose(qa, qb):
    a = ti.Vector([qa[0], qa[1], qa[2]])
    b = ti.Vector([qb[0], qb[1], qb[2]])
    q = qa[3] * b + qb[3] * a + a.cross(b)
    return ti.Vector([q[0], q[1], q[2], qa[3] * qb[3] - a.dot(b)])

@ti.func
def decompose(q):
    a = q[3]
    if (a < -1.0):
        a = -1.0
    if (a > 1.0):
        a = 1.0
    return 2.0 * ti.acos(a)

@ti.func
def rotate(a, q):
    w = ti.Vector([a[0], a[1], a[2], 0.0])
    q_ = ti.Vector([-q[0], -q[1], -q[2], q[3]])
    w_ = compose(compose(q, w), q_)
    return ti.Vector([w_[0], w_[1], w_[2]])

def npCompose(qa, qb):
    a = np.array([qa[0], qa[1], qa[2]])
    b = np.array([qb[0], qb[1], qb[2]])
    q = qa[3] * b + qb[3] * a + np.cross(a, b)
    return np.array([q[0], q[1], q[2], qa[3] * qb[3] - a.dot(b)])

def npRotate(a, q):
    w = np.array([a[0], a[1], a[2], 0.0])
    q_ = np.array([-q[0], -q[1], -q[2], q[3]])
    w_ = npCompose(npCompose(q, w), q_)
    return np.array([w_[0], w_[1], w_[2]])

@ti.func
def makeRotation(theta, axis):
    c = ti.cos(0.5 * theta)
    s = ti.sin(0.5 * theta)
    return ti.Vector([s * axis[0], s * axis[1], s * axis[2], c])

@ti.func
def normalize(a):
    b = a
    if a.norm() > 0.0:
        b = a / a.norm()
    return b

@ti.data_oriented
class SolverBdem3(SolverBase3):
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
        self.rotation = ti.Vector.field(4, ti.f32)
        self.angularVelocity = ti.Vector.field(3, ti.f32)
        self.angularVelocityMid = ti.Vector.field(3, ti.f32)
        self.torsion = ti.Vector.field(3, ti.f32)
        self.bondsLength = ti.field(ti.f32)
        self.bondsSigma = ti.field(ti.f32)
        self.bondsTao = ti.field(ti.f32)
        self.bondsDirection = ti.Vector.field(3, ti.f32)
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
                self.rotation[i] = [0.0, 0.0, 0.0, 1.0]
                # self.mass[i] *= 0.1
                self.momentOfInertia[i] = 2.0 / 5.0 * self.mass[i] * (self.radius[i]**2)
   
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
                f = ti.Vector([0.0, 0.0, 0.0])
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
                    xiS = ti.Vector([0.0, 0.0, 0.0])
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
                vi = self.velocity[i]
                wi = self.angularVelocity[i]
                force = ti.Vector([0.0, 0.0, 0.0])
                torsion = ti.Vector([0.0, 0.0, 0.0])
                sigmaSum = 0.0
                sigmaMax = 0.0
                taoSum = 0.0
                taoMax = 0.0
                sigmaCount = 0
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
                    s0 = np.pi * r0 * r0
                    I0 = 0.25 * np.pi * r0 * r0 * r0 * r0
                    J0 = 2.0 * I0
                    l = (pj - pi).norm()
                    dl = l - l0
                    n = (pj - pi) / l

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
                    dirI = normalize(rotate(d, qi))
                    dirJ = normalize(rotate(d, qj))
                    dij = (dirI + dirJ) * 0.5
                    dn = dij.dot(n)
                    ds = (dij - dn * n).norm(); 
                    thetaij = ti.atan2(ds, dn)
                    t = normalize(dij - dn * n)
                    ft = -kt * thetaij * t
                    vt = (vj - vi) - vn
                    ftDamp = gammat * vt
                    # bend
                    qi_ = ti.Vector([-qi[0], -qi[1], -qi[2], qi[3]])
                    qij = normalize(compose(qj, qi_))
                    theta = decompose(qij)
                    bij = normalize(ti.Vector([qij[0], qij[1], qij[2]])) * theta
                    bn = bij.dot(n) * n
                    bt = bij - bn
                    ml = km * bn
                    mt = km * bt
                    wij = wi - wj
                    wn = wij.dot(n) * n
                    wt = wij - wn
                    mlDamp = -gammam * wn
                    mtDamp = -gammam * wt
                    # fracture
                    sigma = (fn + fnDamp).norm() / s0 + (mt + mtDamp).norm() * r0 / I0
                    tao = (ft + ftDamp).norm() / s0 + (ml + mlDamp).norm() * r0 / J0
                    sigmaSum += sigma
                    taoSum += tao
                    sigmaCount += 1
                    sigmaMax = ti.max(sigmaMax, sigma / sigma0)
                    taoMax = ti.max(taoMax, tao / tao0)
                    if (((dl > 0.0) and sigma > sigma0) or tao > tao0):
                        self.bondsState[idx] = self.scene.BOND_BROKEN
                        # print(i, j, ' | ',sigma, tao, thetaij, ' | ', fn.norm(), ft.norm(), ml.norm(), mt.norm())
                    force += fn + ft
                    force += fnDamp + ftDamp
                    torsion += ml + mt
                    torsion += mlDamp + mtDamp
                    torsion += (ft + ftDamp).cross(-0.5 * l * n)
                self.force[i] += force
                self.torsion[i] += torsion
                if sigmaCount == 0:
                    self.color[i] = [0.0, 0.0, 0.0]
                else:
                    self.color[i] = ti.Vector([0.5 * (sigmaMax + taoMax), 0.0, 0.0])
                    # self.color[i] = ti.Vector([sigmaSum / sigmaCount / self.sigmac / self.kn, 0.0, 0.0])
                    # self.color[i] = ti.Vector([taoSum / sigmaCount / self.tauc / self.kt, 0.0, 0.0])
            
    @ti.kernel
    def updatePosition(self, dt: ti.f64):
        for i in self.position:
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            self.torsion[i] = ti.Vector([0.0, 0.0, 0.0])
            self.position[i] += self.velocityMid[i] * dt; 
            wi = self.angularVelocityMid[i]
            wt = makeRotation(wi.norm() * dt, normalize(wi))
            self.rotation[i] = normalize(compose(wt, self.rotation[i]))

            li = self.label[i]
            if li != self.scene.FLUID:
                speed = 1.0 * 2.0 * np.pi
                pi = self.position[i]
                if pi[0] > 0.0:
                    speed = -speed
                theta = speed * self.t[0]
                c = ti.cos(theta)
                s = ti.sin(theta)
                tx = self.positionInitial[i][2]
                ty = self.positionInitial[i][1]
                self.position[i][2] = c * tx - s * ty
                self.position[i][1] = s * tx + c * ty
                axis = ti.Vector([speed, 0.0, 0.0])
                self.velocity[i] = -axis.cross(self.position[i])

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
        self.computeLocalDamping(dt)
        # self.computeBoundary(dt)

        self.updateVelocity(dt)