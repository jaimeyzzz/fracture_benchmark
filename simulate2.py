import numpy as np
import sys
import taichi as ti

from neighbor_search2 import NeighborSearch2
from scene import Scene
from solver_mass_spring2 import SolverMassSpring2
from solver_peridynamics2 import SolverPeridynamics2

#ti.init(debug=True, log_level=ti.TRACE)
ti.init(arch=ti.cpu, debug=True)

NUM_FRAMES = 400
SCENE_FOLDER = 'scene2'
SCENE_NAME = sys.argv[1]
WINDOW_SIZE = 640

fps = 60.0
cfl = 0.05

scene = Scene(SCENE_FOLDER, SCENE_NAME)

neighborSearch = NeighborSearch2(scene.lowerBound, scene.upperBound, scene.r * 2.0)
solver = SolverPeridynamics2(scene, neighborSearch)
# TAICHI MATERIALIZE
neighborSearch.init()
solver.init()

""" RUN SIMULATION """
gui = ti.GUI('demo', (WINDOW_SIZE, WINDOW_SIZE), background_color=0xFFFFFF)
dt = np.sqrt(np.min(scene.mass[scene.label == scene.FLUID]) / scene.kc) * cfl
print('Sub Timestep : ', dt)
s = np.maximum(1, (int)(1 / fps / dt))
dt = 1 / fps / float(s)


npLoad = np.float32([])
npTime = np.float32([])

frameIdx = 0
while gui.running and frameIdx < NUM_FRAMES:
    for _ in range(s):
        solver.update(dt)

    load = solver.load.to_numpy()[0]
    npLoad = np.append(npLoad, np.linalg.norm(load))
    npTime = np.append(npTime, solver.t[0])

    npPosition = solver.position.to_numpy() + [0.5, 0.5]
    npLabel = solver.label.to_numpy()
    radius = WINDOW_SIZE * scene.r
    gui.circles(npPosition[npLabel == 0], radius=radius, color=0x0088ff)
    gui.circles(npPosition[npLabel != 0], radius=radius, color=0xffaa11)
    # # bonds
    # npBondsIdx = solver.bondsIdx.to_numpy()
    # npBondsNum = solver.bondsNum.to_numpy()
    # npBondsState = solver.bondsState.to_numpy()
    # npBegin = np.array([])
    # npEnd = np.array([])
    # for i in range(npBondsNum.shape[0]):
    #     for idx in range(npBondsNum[i]): 
    #         j = npBondsIdx[i][idx]
    #         if npBondsState[i][idx] != 0: continue
    #         if i < j:
    #             npBegin = np.append(npBegin, npPosition[i])
    #             npEnd = np.append(npEnd, npPosition[j])
    # npBegin = npBegin.reshape((-1, 2))
    # npEnd = npEnd.reshape((-1, 2))
    # gui.lines(npBegin, npEnd, color=0x808080)
    frameIdx += 1
    gui.show(f'output/images/{frameIdx:04d}.jpg')

np.savez('output/data.npz', load=npLoad, time=npTime)