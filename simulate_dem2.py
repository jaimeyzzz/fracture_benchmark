import numpy as np
import taichi as ti

from neighbor_search2 import NeighborSearch2
from peridynamics_solver2 import PeridynamicsSolver2

#ti.init(debug=True, log_level=ti.TRACE)
ti.init(arch=ti.cpu, debug=True)

WINDOW_SIZE = 640

fps = 60.0
cfl = 0.05
r = 0.010
rho = 2.5e3
kn = 1e5
kc = 1e6
gammac = 0.7
us = 0.1
npLowerBound = np.float32([0.0, 0.0])
npUpperBound = np.float32([1.0, 1.0])
npGravity = np.float32([0.0, -9.81])
npPos = np.array([], dtype=np.float32)
npMass = np.array([], dtype=np.float32)
npRadius = np.array([], dtype=np.float32)
npLabels = np.array([], dtype=np.int32)
npVelocity = np.array([], dtype=np.int32)

def addFluid(low, high, vel):
    global npPos, npMass, npLabels, npRadius, npVelocity
    low = np.array(low)
    high = np.array(high)
    vel = np.array(vel)
    size = ((high - low) / r / 2.0).astype(int)
    for xi in range(size[0]):
        for yi in range(size[1]):
            npPos = np.append(npPos, low + np.array([xi, yi]) * r * 2.0)
            npMass = np.append(npMass, rho * r * r * np.pi)
            npRadius = np.append(npRadius, r)
            npLabels = np.append(npLabels, 0)
            npVelocity = np.append(npVelocity, vel)

def addBoundary(low, high, vel):
    global npPos, npMass, npLabels, npRadius, npVelocity
    low = np.array(low)
    high = np.array(high)
    vel = np.array(vel)
    size = ((high - low) / r / 2.0).astype(int) + 1
    for xi in range(size[0]):
        for yi in range(size[1]):
            npPos = np.append(npPos, low + np.array([xi, yi]) * r * 2.0)
            npMass = np.append(npMass, -1.0)
            npRadius = np.append(npRadius, r)
            npLabels = np.append(npLabels, 1)
            npVelocity = np.append(npVelocity, vel)

addBoundary([r, r], [5.0 * r, 1.0 - r], [0.0, 0.0])
addBoundary([1.0 - 5.0 * r, r], [1.0 - r, 1.0 - r], [0.0, 0.0])
addBoundary([r, r], [1.0 - r, 5.0 * r], [0.0, 0.0])
addBoundary([r, 1.0 - 5.0 * r], [1.0 - r, 1.0 - r], [0.0, 0.0])

addBoundary([0.45, 0.7], [0.55, 0.8], [0.0, -0.5])

addFluid([7.0 * r, 0.4], [1.0 - 3.0 * r, 0.5], [0.0, 0.0])

npPos = npPos.reshape((-1,2)).astype(np.float32)
npVelocity = npVelocity.reshape((-1,2)).astype(np.float32)
npMass = npMass.reshape((-1,)).astype(np.float32)
npRadius = npRadius.reshape((-1,)).astype(np.float32)
npLabels = npLabels.reshape((-1,)).astype(np.int32)

numParticles = npPos.shape[0]

neighborSearch = NeighborSearch2(npLowerBound, npUpperBound, r * 2.0)
bondSearch = NeighborSearch2(npLowerBound, npUpperBound, r * 4.0)
solver = PeridynamicsSolver2(numParticles, kn, kc, gammac, us, r * 4.0)

neighborSearch.init()
bondSearch.init()
solver.init(npPos, npVelocity, npMass, npLabels, npRadius, neighborSearch, bondSearch)

""" RUN SIMULATION """
gui = ti.GUI('demo', (WINDOW_SIZE, WINDOW_SIZE), background_color=0xFFFFFF)
dt = np.sqrt(rho * np.pi * r * r / kc) * cfl
print('Sub Timestep : ', dt)
s = np.maximum(1, (int)(1 / fps / dt))
dt = 1 / fps / float(s)

frameIdx = 0
while gui.running:
    for _ in range(s):
        solver.update(dt)

    npPos = solver.position.to_numpy()
    npMass = solver.mass.to_numpy()
    radius = WINDOW_SIZE * r
    gui.circles(npPos[npMass > 0.0], radius=radius, color=0x0088ff)
    gui.circles(npPos[npMass < 0.0], radius=radius, color=0xffaa11)
    frameIdx += 1
    gui.show(f'output/{frameIdx:04d}.jpg')