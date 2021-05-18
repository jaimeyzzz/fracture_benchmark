import numpy as np
import os
import sys
import taichi as ti
import time

from neighbor_search2 import NeighborSearch2
from scene import Scene
from solver_bdem2 import SolverBdem2
from solver_dem2 import SolverDem2
from solver_mass_spring2 import SolverMassSpring2
from solver_mpm2 import SolverMpm2
from solver_peridynamics2 import SolverPeridynamics2

#ti.init(debug=True, log_level=ti.TRACE)
ti.init(arch=ti.cpu, debug=True)

SCENE_FOLDER = 'scene2'
SCENE_NAME = sys.argv[1]
SOLVER_NAME = sys.argv[2]
NUM_FRAMES = int(sys.argv[3])
M = 6
WINDOW_SIZE = 640

scene = Scene(SCENE_FOLDER, SCENE_NAME)
outputDir = 'output/images/{}_{}2/'.format(SCENE_NAME, SOLVER_NAME)
try:
    os.mkdir(outputDir)
except OSError as error:
    print(error)   

neighborSearch = NeighborSearch2(scene.lowerBound, scene.upperBound, scene.r * 2.0)
solver = None
if SOLVER_NAME == 'bdem':
    scene.cfl *= 0.5
    solver = SolverBdem2(scene, neighborSearch)
elif SOLVER_NAME == 'dem':
    solver = SolverDem2(scene, neighborSearch)
elif SOLVER_NAME == 'mass_spring':
    solver = SolverMassSpring2(scene, neighborSearch)
elif SOLVER_NAME == 'peridynamics':
    scene.kn = 1e4
    scene.kt = 1e4
    solver = SolverPeridynamics2(scene, neighborSearch)
    M = 42
elif SOLVER_NAME == 'mpm':
    scene.cfl *= 1.0
    scene.fps = 60.0
    solver = SolverMpm2(scene, neighborSearch)
    
# TAICHI MATERIALIZE
neighborSearch.init()
solver.init()

""" RUN SIMULATION """
gui = ti.GUI('demo', (WINDOW_SIZE, WINDOW_SIZE), background_color=0xFFFFFF)
k = np.max([scene.kc, scene.kn * scene.rMax * M * M])
print('Stiffness : ', k)
dt = np.sqrt(0.5 * scene.mMin / k) * scene.cfl
print('Sub Timestep : ', dt)
s = np.maximum(1, (int)(1 / scene.fps / dt))
dt = 1 / scene.fps / float(s)

npLoad = np.float32([])
npTime = np.float32([])



frameIdx = 0
while gui.running and frameIdx < NUM_FRAMES:
    start = time.time()
    for _ in range(s):
        solver.update(dt)

        load = solver.load.to_numpy()[0]
        npLoad = np.append(npLoad, np.linalg.norm(load))
        npTime = np.append(npTime, solver.t[0])
    end = time.time()
    print('Process frame', frameIdx, 'for', end - start)
    print(load, solver.t[0])

    npPosition = solver.position.to_numpy() / [1.0, 1.0] + [0.5, 0.5]
    npLabel = solver.label.to_numpy()
    radius = WINDOW_SIZE * scene.r
    gui.circles(npPosition[npLabel == 0], radius=radius, color=0xffaa11) # 0088ff
    gui.circles(npPosition[npLabel != 0], radius=radius, color=0x000000)
    # # bonds
    # npBondsIdx = solver.bondsIdx.to_numpy()
    # npBondsAccum = solver.bondsAccum.to_numpy()
    # npBondsState = solver.bondsState.to_numpy()
    # npBegin = np.array([])
    # npEnd = np.array([])
    # for i in range(npBondsAccum.shape[0] - 1):
    #     for idx in range(npBondsAccum[i], npBondsAccum[i + 1]): 
    #         j = npBondsIdx[idx]
    #         if npBondsState[idx] != 0: continue
    #         if i < j:
    #             npBegin = np.append(npBegin, npPosition[i])
    #             npEnd = np.append(npEnd, npPosition[j])
    # npBegin = npBegin.reshape((-1, 2))
    # npEnd = npEnd.reshape((-1, 2))
    # gui.lines(npBegin, npEnd, color=0x000)
    frameIdx += 1
    gui.show(outputDir+f'{frameIdx:04d}.jpg')

np.savez('output/{}_{}2.npz'.format(SCENE_NAME, SOLVER_NAME), load=npLoad, time=npTime)