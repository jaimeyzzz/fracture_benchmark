import numpy as np
import taichi as ti

from neighbor_search2 import NeighborSearch2
from solver_base2 import SolverBase2

@ti.data_oriented
class SolverMpm2(SolverBase2):
    GRID_SIZE = 32
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
        self.sigmaF = 2.0e6
        self.deleteThreshold = 1.0
        self.mu = self.E / 2.0 * (1.0 + self.nu)    
        self.lamb = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.kappa = 2.0 / 3.0 * self.mu + self.lamb

        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1+self.nu) * (1 - 2 * self.nu)) # Lame parameters
        self.particleVol = (self.gridSpacing * 0.5)**2
        self.particleRho = 1.0e3
        self.particleMass = self.particleVol * self.particleRho

        self.gridMass = ti.field(ti.f32)
        self.gridVelocity = ti.Vector.field(2, ti.f32)
        self.C = ti.Matrix.field(2, 2, ti.f32)
        self.gD = ti.Matrix.field(2, 2, ti.f32)
        self.Jp = ti.field(ti.f32)
        self.gridOrigin = ti.Vector.field(2, ti.f32, 1)
        self.phaseC = ti.field(ti.f32)
        self.phaseG = ti.field(ti.f32)
        self.phaseK = 0.001

        ti.root.dense(ti.i, self.N).place(self.C)
        ti.root.dense(ti.i, self.N).place(self.gD)
        ti.root.dense(ti.i, self.N).place(self.Jp)
        ti.root.dense(ti.i, self.N).place(self.phaseC)
        ti.root.dense(ti.i, self.N).place(self.phaseG)
        ti.root.dense(ti.ij, (self.GRID_SIZE, self.GRID_SIZE)).place(self.gridMass)
        ti.root.dense(ti.ij, (self.GRID_SIZE, self.GRID_SIZE)).place(self.gridVelocity)
   
    def init(self):
        super().init()
        self.initProperties()
        self.gridOrigin.from_numpy(self.scene.lowerBound.reshape(1,2))

    @ti.func
    def cofactorMatrix(self, F):
        A = ti.Matrix.zero(ti.f32,2,2)
        A[0, 0] = F[1, 1]
        A[1, 0] = -F[0, 1]
        A[0, 1] = -F[1, 0]
        A[1, 1] = F[0, 0]
        return A

    @ti.func
    def contractMatrices(self, F1, F2):
        result = 0.0
        for i, j in ti.static(ti.ndrange(2, 2)):
            result += F1[i, j] * F2[i, j]
        return result

    @ti.kernel
    def initProperties(self):
        for i in self.position:
            li = self.label[i]
            self.gD[i] = ti.Matrix.identity(ti.f32, 2)
            self.Jp[i] = 1.0
            if li == self.scene.FLUID:
                self.mass[i] = self.particleMass
                self.phaseC[i] = 1.0
            else:
                self.mass[i] = self.particleMass #* 1e4

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
            phasei = self.phaseC[i]
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
            # # NeoHookean
            # F = self.gD[i]
            # J = F.determinant()
            # I2 = ti.Matrix.identity(float, 2)
            # stress = self.mu * (F @ F.transpose() - I2) + self.lambda_0 * ti.log(J) * I2
            # # NeoHookeanBorden Model
            # Flocal = F
            # gradDv = Ci
            # dF = gradDv @ Flocal
            # g = self.phaseG[i]
            # FinvT = (1.0 / J) * self.cofactorMatrix(F)
            # deltaJ = J * self.contractMatrices(FinvT, dF)
            # FcontractF = self.contractMatrices(F, F)
            # deltaFinvT = -FinvT @ dF.transpose() @ FinvT
            # #print(dF, dP, FinvT, deltaJ, FcontractF, deltaFinvT)
            # a = -1.0 / 2.0
            # muJ2a = self.mu_0 * ti.pow(J, 2.0 * a)
            # prime = self.kappa / 2.0 * (J - 1.0 / J)
            # prime2 = self.kappa / 2 * (1.0 + 1.0 / (J * J))
            # dP_dev = 2 * a * muJ2a * self.contractMatrices(FinvT, dF) * (a * FcontractF * FinvT + F)
            # dP_dev += muJ2a * (2 * a * self.contractMatrices(F, dF) * FinvT + a * FcontractF * deltaFinvT + dF)
            # dP_vol = deltaJ * prime * FinvT + J * deltaJ * prime2 * FinvT + J * prime * deltaFinvT
            # dP = ti.Matrix.zero(ti.f32, 2, 2)
            # if (J >= 1.0):
            #     dP = g * (dP_dev + dP_vol)
            # else:
            #     dP = g * dP_dev + dP_vol
            # # print(dP, dP_dev, dP_vol)
            # # stress = phasei * dP @ Flocal.transpose()
            # V_p^0 * dP * F^T
            stress = (-dt * self.particleVol * 4 / dx / dx) * stress
            affine = stress + mi * Ci
            for j, k in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[j][0] * w[k][1]
                self.gridVelocity[base + offset] += weight * (mi * vi + affine @ dpos)
                self.gridMass[base + offset] += weight * mi

    @ti.kernel
    def computeExternal(self, dt: ti.f32):
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

    @ti.kernel
    def deleteParticles(self, dt: ti.f32):
        for i in self.position:
            li = self.label[i]
            ci = self.phaseC[i]
            F = self.gD[i]
            J = F.determinant()
            if J < 1.1: continue
            if li == self.scene.FLUID and ci < self.deleteThreshold:
                print('***', J)
                self.position[i] = [-1.0, -1.0]
                self.label[i] = self.scene.BOUNDARY

    @ti.kernel
    def solvePhase(self, dt: ti.f32):
        for i in self.label:
            li = self.label[i]
            if li == self.scene.FLUID:
                ci = self.phaseC[i]
                F = self.gD[i]
                J = F.determinant()
                tau = ti.Matrix.zero(ti.f32,2,2)
                g = 1.0
                # tau = kirchhoff
                B = F @ F.transpose()
                devB = B - ti.Matrix.identity(ti.f32,2) * 1.0 / 2.0 * B.trace()
                tauDev = self.mu * ti.pow(J, -2.0 / 2.0) * devB
                prime = self.kappa / 2.0 * (J - 1.0 / J)
                tauVol = J * prime * ti.Matrix.identity(ti.f32,2)
                if (J >= 1.0):
                    tau = g * (tauDev + tauVol)
                else:
                    tau = g * tauDev + tauVol
                cauchy = tau / J
                eigenValues, eifenVectors = ti.eig(cauchy, ti.f32)
                # print(eigenValues)
                sigmaMax = eigenValues[0, 0]
                newCi = ci
                if (sigmaMax > self.sigmaF):
                    # print('###', self.sigmaF / sigmaMax)
                    newCi = ti.min(ci, self.sigmaF / sigmaMax)
                self.phaseC[i] = newCi
                self.phaseG[i] = newCi * newCi * (1 - self.phaseK) + self.phaseK

    def update(self, dt):
        self.updateTime(dt)

        self.deleteParticles(dt)

        self.updatePosition(dt)
        # self.updateNeighbors()
        self.load.fill(0)

        self.particleToGrid(dt)
        self.solvePhase(dt)
        self.computeExternal(dt)
        self.gridToParticle(dt)

        # self.computeCollision(dt)
        # self.computeGravity(dt)
        # self.updateVelocity(dt)

        # self.computeBoundary(dt)