import numpy as np
import taichi as ti

from neighbor_search2 import NeighborSearch2
from solver_base2 import SolverBase2

@ti.data_oriented
class SolverMpm2(SolverBase2):
    GRID_SIZE = 64
    def __init__(self, scene, neighborSearch):
        super().__init__(scene, neighborSearch)
        self.kn = scene.kn
        self.kt = scene.kt
        self.kc = scene.kc
        self.gammac = scene.gammac
        self.us = scene.us
        self.h = scene.h
        self.gridSpacing = np.max(scene.upperBound - scene.lowerBound) / self.GRID_SIZE
        self.E = 1.0e7
        self.nu = 0.25
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1+self.nu) * (1 - 2 * self.nu)) # Lame parameters
        self.gridMass = ti.field(ti.f32)
        self.gridVelocity = ti.Vector.field(2, ti.f32)
        self.C = ti.Matrix.field(2, 2, ti.f32)
        self.gD = ti.Matrix.field(2, 2, ti.f32)
        self.Jp = ti.field(ti.f32)
        self.gridOrigin = ti.Vector.field(2, ti.f32, 1)
        self.particleVol = (self.gridSpacing * 0.5)**2
        self.particleRho = 1.0e3
        self.particleMass = self.particleVol * self.particleRho

        ti.root.dense(ti.i, self.N).place(self.C)
        ti.root.dense(ti.i, self.N).place(self.gD)
        ti.root.dense(ti.i, self.N).place(self.Jp)
        ti.root.dense(ti.ij, (self.GRID_SIZE, self.GRID_SIZE)).place(self.gridMass)
        ti.root.dense(ti.ij, (self.GRID_SIZE, self.GRID_SIZE)).place(self.gridVelocity)
   
    def init(self):
        super().init()
        self.initProperties()
        self.gridOrigin.from_numpy(self.scene.lowerBound.reshape(1,2))

    @ti.kernel
    def initProperties(self):
        for i in self.position:
            li = self.label[i]
            self.gD[i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[i] = 1
            if li == self.scene.FLUID:
                self.mass[i] = self.particleMass
            else:
                self.mass[i] = self.particleMass #* 1e4


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
    def particleToGrid(self, dt: ti.f32):
        for i, j in self.gridMass:
            self.gridMass[i, j] = 0.0
            self.gridVelocity[i, j] = [0.0, 0.0]
        for i in self.position:
            li = self.label[i]
            if li != self.scene.FLUID: continue
            mi = self.mass[i]
            xi = self.position[i]
            vi = self.velocity[i]
            Ci = self.C[i]
            dx = self.gridSpacing
            base = ((xi - self.gridOrigin[0])/ dx - 0.5).cast(int)
            fx = (xi - self.gridOrigin[0]) / dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.gD[i] = (ti.Matrix.identity(float, 2) + dt * Ci) @ self.gD[i] # deformation gradient update
            h = ti.exp(10 * (1.0 - self.Jp[i])) # Hardening coefficient: snow gets harder when compressed
            h = 1.0
            mu, la = self.mu_0 * h, self.lambda_0 * h
            U, sig, V = ti.svd(self.gD[i])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                self.Jp[i] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            stress = 2 * mu * (self.gD[i] - U @ V.transpose()) @ self.gD[i].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = (-dt * self.particleVol * 4 / dx / dx) * stress
            affine = stress + mi * Ci
            for j, k in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[j][0] * w[k][1]
                self.gridVelocity[base + offset] += weight * (mi * vi + affine @ dpos)
                self.gridMass[base + offset] += weight * mi
        cy = 0.16 - self.t[0] * 0.1
        for i, j in self.gridMass:
            dx = self.gridSpacing
            if self.gridMass[i, j] > 0: # No need for epsilon here
                self.gridVelocity[i, j] = (1 / self.gridMass[i, j]) * self.gridVelocity[i, j] # Momentum to velocity
                self.gridVelocity[i, j][1] -= dt * 9.81 # gravity

                dist = ti.Vector([i * dx, j * dx]) + self.gridOrigin[0] - [0.28, -0.15]
                if dist.norm() < 0.07:
                    dist = dist.normalized()
                    self.gridVelocity[i, j] -= dist * min(0, self.gridVelocity[i, j].dot(dist))
                dist = ti.Vector([i * dx, j * dx]) + self.gridOrigin[0] - [-0.28, -0.15]
                if dist.norm() < 0.07:
                    dist = dist.normalized()
                    self.gridVelocity[i, j] -= dist * min(0, self.gridVelocity[i, j].dot(dist))
                dist = ti.Vector([i * dx, j * dx]) + self.gridOrigin[0] - [0.0, cy]
                if dist.norm() < 0.07:
                    dist = dist.normalized()
                    deltaV = -dist * min(0, self.gridVelocity[i, j].dot(dist)) + [0.0, -0.1]
                    self.gridVelocity[i, j] += deltaV
                    self.load[0] += deltaV * self.gridMass[i, j] / dt

                if i < 3 and self.gridVelocity[i, j][0] < 0:          self.gridVelocity[i, j][0] = 0 # Boundary conditions
                if i > self.GRID_SIZE - 3 and self.gridVelocity[i, j][0] > 0: self.gridVelocity[i, j][0] = 0
                if j < 3 and self.gridVelocity[i, j][1] < 0:          self.gridVelocity[i, j][1] = 0
                if j > self.GRID_SIZE - 3 and self.gridVelocity[i, j][1] > 0: self.gridVelocity[i, j][1] = 0

    @ti.kernel
    def gridToParticle(self, dt: ti.f32):
        for i in self.position:
            li = self.label[i]
            if li != self.scene.FLUID: continue
            mi = self.mass[i]
            xi = self.position[i]
            vi = self.velocity[i]
            dx = self.gridSpacing
            base = ((xi - self.gridOrigin[0]) / dx - 0.5).cast(int)
            fx = (xi - self.gridOrigin[0]) / dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            newVi = ti.Vector.zero(float, 2)
            newCi = ti.Matrix.zero(float, 2, 2)
            for j, k in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([j, k])
                dpos = (offset.cast(float) - fx)
                weight = w[j].x * w[k].y
                gV = self.gridVelocity[base + offset]
                newVi += weight * gV               
                newCi += 4 * weight * gV.outer_product(dpos) / dx
            self.velocity[i] = newVi
            self.position[i] += dt * self.velocity[i]
            self.C[i] = newCi
            
    @ti.kernel
    def updatePosition(self, dt: ti.f32):
        for i in self.position:
            li = self.label[i]
            if li == self.scene.FLUID: continue
            self.force[i] = ti.Vector([0.0, 0.0])
            self.position[i] += self.velocityMid[i] * dt

    def update(self, dt):
        self.updateTime(dt)

        self.updatePosition(dt)
        # self.updateNeighbors()
        self.load.fill(0)

        self.particleToGrid(dt)
        self.gridToParticle(dt)

        # self.computeCollision(dt)
        # self.computeGravity(dt)
        # self.updateVelocity(dt)

        # self.computeBoundary(dt)