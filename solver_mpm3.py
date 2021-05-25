import numpy as np
import taichi as ti

from neighbor_search3 import NeighborSearch3
from solver_base3 import SolverBase3

@ti.data_oriented
class SolverMpm3(SolverBase3):
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
        self.E = self.kn
        self.nu = 0.25
        self.sigmaF = self.kn * self.scene.sigma
        self.deleteThreshold = 0.01
        self.mu = self.E / 2.0 * (1.0 + self.nu)    
        self.lamb = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.kappa = 2.0 / 3.0 * self.mu + self.lamb

        # self.particleVol = (self.gridSpacing * 0.5)**3
        self.particleVol = (self.scene.rMin)**3 * np.pi * 4.0 / 3.0
        self.particleRho = 2600.0
        self.particleMass = self.particleVol * self.particleRho

        self.gridMass = ti.field(ti.f32)
        self.gridVelocity = ti.Vector.field(3, ti.f32)
        self.C = ti.Matrix.field(3, 3, ti.f32)
        self.F = ti.Matrix.field(3, 3, ti.f32)
        self.cauchy = ti.Matrix.field(3, 3, ti.f32)
        self.Jp = ti.field(ti.f32)
        self.gridOrigin = ti.Vector.field(3, ti.f32, 1)
        self.phaseC = ti.field(ti.f32)
        self.phaseG = ti.field(ti.f32)
        self.sigmaMax = ti.field(ti.f32)
        self.phaseK = 0.001
        self.distance = ti.Vector.field(3, ti.f32)

        ti.root.dense(ti.i, self.N).place(self.C)
        ti.root.dense(ti.i, self.N).place(self.F)
        ti.root.dense(ti.i, self.N).place(self.Jp)
        ti.root.dense(ti.i, self.N).place(self.phaseC)
        ti.root.dense(ti.i, self.N).place(self.phaseG)
        ti.root.dense(ti.i, self.N).place(self.sigmaMax)
        ti.root.dense(ti.i, self.N).place(self.cauchy)
        ti.root.dense(ti.i, self.N).place(self.distance)
        ti.root.dense(ti.ijk, (self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)).place(self.gridMass)
        ti.root.dense(ti.ijk, (self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)).place(self.gridVelocity)
   
    def init(self):
        super().init()
        self.initProperties()
        self.gridOrigin.from_numpy(self.scene.lowerBound.reshape(1,3))

    @ti.kernel
    def initProperties(self):
        for i in self.position:
            li = self.label[i]
            self.F[i] = ti.Matrix.identity(ti.f32, 3)
            if li == self.scene.FLUID:
                self.mass[i] = self.particleMass
                self.phaseC[i] = 1.0
                self.phaseG[i] = 1.0
            else:
                self.mass[i] = self.particleMass #* 1e4
                self.distance[i] = self.position[i]

    @ti.kernel
    def particleToGrid(self, dt: ti.f32):
        for i, j, k in self.gridMass:
            self.gridMass[i, j, k] = 0.0
            self.gridVelocity[i, j, k] = [0.0, 0.0, 0.0]
        for i in self.position:
            li = self.label[i]
            # if li != self.scene.FLUID: continue
            mi = self.mass[i]
            xi = self.position[i]
            vi = self.velocity[i]
            Ci = self.C[i]
            phasei = self.phaseC[i]
            g = self.phaseG[i]
            dx = self.gridSpacing
            base = ((xi - self.gridOrigin[0])/ dx - 0.5).cast(int)
            fx = (xi - self.gridOrigin[0]) / dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.F[i] = (ti.Matrix.identity(float, 3) + dt * Ci) @ self.F[i] # deformation gradient update
            F = self.F[i]
            J = self.F[i].determinant()

            tau = ti.Matrix.zero(ti.f32,3,3)
            B = F @ F.transpose()
            devB = B - ti.Matrix.identity(ti.f32,3) * 1.0 / 3.0 * B.trace()
            tauDev = self.mu * ti.pow(J, -2.0 / 3.0) * devB
            prime = self.kappa / 2.0 * (J - 1.0 / J)
            tauVol = J * prime * ti.Matrix.identity(ti.f32,3)
            tau = ti.Matrix.zero(ti.f32,3,3)
            if (J >= 1.0):
                tau = g * (tauDev + tauVol)
            else:
                tau = g * tauDev + tauVol
            stress = tau.transpose()
            stress = (-dt * self.particleVol * 4 / dx / dx) * stress
            affine = stress + mi * Ci
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))): # Loop over 3x3 grid node neighborhood
                dpos = (offset.cast(float) - fx) * dx
                weight = 1.0
                for k in ti.static(range(3)):
                    weight *= w[offset[k]][k]
                self.gridVelocity[base + offset] += weight * (mi * vi + affine @ dpos)
                self.gridMass[base + offset] += weight * mi

    @ti.kernel
    def computeExternal(self, dt: ti.f32):
        for I in ti.grouped(self.gridMass):
            dx = self.gridSpacing
            if self.gridMass[I] > 0: # No need for epsilon here
                self.gridVelocity[I] = (1 / self.gridMass[I]) * self.gridVelocity[I] # Momentum to velocity
                self.gridVelocity[I][1] -= dt * 9.81 #self.scene.gravity

                # if I[0] < 1 and self.gridVelocity[I][0] < 0:          self.gridVelocity[I][0] = 0 # Boundary conditions
                # if I[0] > self.GRID_SIZE - 1 and self.gridVelocity[I][0] > 0: self.gridVelocity[I][0] = 0
                # if I[1] < 1 and self.gridVelocity[I][1] < 0:          self.gridVelocity[I][1] = 0
                # if I[1] > self.GRID_SIZE - 1 and self.gridVelocity[I][1] > 0: self.gridVelocity[I][1] = 0
                # if I[2] < 1 and self.gridVelocity[I][2] < 0:          self.gridVelocity[I][2] = 0
                # if I[2] > self.GRID_SIZE - 1 and self.gridVelocity[I][2] > 0: self.gridVelocity[I][2] = 0

    @ti.kernel
    def gridToParticle(self, dt: ti.f32):
        for i in self.position:
            li = self.label[i]
            if li == self.scene.FLUID:
                mi = self.mass[i]
                xi = self.position[i]
                vi = self.velocity[i]
                dx = self.gridSpacing
                base = ((xi - self.gridOrigin[0]) / dx - 0.5).cast(int)
                fx = (xi - self.gridOrigin[0]) / dx - base.cast(float)
                w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
                newVi = ti.Vector.zero(float, 3)
                newCi = ti.Matrix.zero(float, 3, 3)
                for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    dpos = (offset - fx) * dx
                    weight = 1.0
                    for k in ti.static(range(3)):
                        weight *= w[offset[k]][k]
                    g_v = self.gridVelocity[base + offset]
                    newVi += weight * g_v
                    newCi += 4 * weight * g_v.outer_product(dpos) / dx**2
                self.velocity[i] = newVi
                self.position[i] += dt * self.velocity[i]
                self.C[i] = newCi
            else:
                speed = 1.0 * 2.0 * np.pi
                pi = self.position[i]
                if pi[0] > 0.0:
                    speed = -speed
                theta = speed * self.t[0]
                c = ti.cos(theta)
                s = ti.sin(theta)
                tx = self.distance[i][2]
                ty = self.distance[i][1]
                self.position[i][2] = c * tx - s * ty
                self.position[i][1] = s * tx + c * ty
                axis = ti.Vector([speed, 0.0, 0.0])
                self.velocity[i] = -axis.cross(self.position[i])
                # self.position[i] += dt * self.velocity[i]
            
    @ti.kernel
    def updatePosition(self, dt: ti.f32):
        for i in self.position:
            li = self.label[i]
            if li == self.scene.FLUID: continue
            speed = 1.0 * 2.0 * np.pi
            pi = self.position[i]
            if pi[0] > 0.0:
                speed = -speed
            theta = speed * dt
            tx = pi[2]
            ty = pi[1]
            c = ti.cos(theta)
            s = ti.sin(theta)
            # self.position[i][2] = c * tx - s * ty
            # self.position[i][1] = s * tx + c * ty
            axis = ti.Vector([speed, 0.0, 0.0])
            self.velocity[i] = -axis.cross(self.position[i])
            # self.velocity[i] = -axis / 50.0

    @ti.kernel
    def deleteParticles(self, dt: ti.f32):
        for i in self.position:
            li = self.label[i]
            ci = self.phaseC[i]
            F = self.F[i]
            J = F.determinant()
            if li == self.scene.FLUID and ci < self.deleteThreshold:
                # print('###', J)
                self.position[i] = [-1.0, -1.0, -1.0]
                self.label[i] = self.scene.BOUNDARY

    @ti.kernel
    def solvePhase(self, dt: ti.f32):
        for i in self.label:
            li = self.label[i]
            if li == self.scene.FLUID:
                ci = self.phaseC[i]
                F = self.F[i]
                J = F.determinant()
                tau = ti.Matrix.zero(ti.f32,3,3)
                g = 1.0
                # tau = kirchhoff
                B = F @ F.transpose()
                devB = B - ti.Matrix.identity(ti.f32,3) * 1.0 / 3.0 * B.trace()
                tauDev = self.mu * ti.pow(J, -2.0 / 3.0) * devB
                prime = self.kappa / 2.0 * (J - 1.0 / J)
                tauVol = J * prime * ti.Matrix.identity(ti.f32,3)
                # print(ti.Matrix.identity(ti.f32,3))
                if (J >= 1.0):
                    tau = g * (tauDev + tauVol)
                else:
                    tau = g * tauDev + tauVol
                # self.cauchy[i] = tau / J
                cauchy = tau / J
                # eigenValues = np.linalg.eig(cauchy)
                # eigenValues, eigenVectors = ti.eig(cauchy, ti.f32)
                _, sig, _ = ti.svd(cauchy)
                a, b, c = sig[0], sig[1], sig[2]
                sigmaMax = ti.max(a, ti.max(b, c)) #ti.max(sig[0], sig[1], sig[2])
                # print('sigmaMax', sigmaMax)
                newCi = ci
                if (sigmaMax > self.sigmaF):
                    # print('###', J, self.sigmaF / sigmaMax)
                    newCi = ti.min(ci, self.sigmaF / sigmaMax)
                self.phaseC[i] = newCi
                # self.phaseG[i] = newCi * newCi * (1 - self.phaseK) + self.phaseK
                self.color[i] = ti.Vector([sigmaMax / self.sigmaF, 0.0, 0.0])
                # self.color[i] = ti.Vector([newCi, 0.0, 0.0])

    def update(self, dt):
        self.updateTime(dt)

        # self.deleteParticles(dt)

        # self.updatePosition(dt)
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