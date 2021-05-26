import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import taichi as ti
import time

from neighbor_search2 import NeighborSearch2
from scene import Scene
from solver_base2 import SolverBase2
from solver_bdem2 import SolverBdem2
from solver_dem2 import SolverDem2
from solver_mass_spring2 import SolverMassSpring2
from solver_mpm2 import SolverMpm2
from solver_peridynamics2 import SolverPeridynamics2

#ti.init(debug=True, log_level=ti.TRACE)
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f32)

SCENE_FOLDER = 'scene2'
SCENE_NAME = sys.argv[1]
SOLVER_NAME = sys.argv[2]
NUM_FRAMES = int(sys.argv[3])
WINDOW_SIZE = 640
M = 1

scene = Scene(SCENE_FOLDER, SCENE_NAME)
outputDir = 'output/images/{}_{}2/'.format(SCENE_NAME, SOLVER_NAME)
try:
    os.mkdir(outputDir)
except OSError as error:
    print(error) 

neighborSearch = NeighborSearch2(scene.lowerBound, scene.upperBound, scene.r * 2.0)
solver = SolverBase2(scene, neighborSearch)
# EIGENS
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs

data = np.ones(solver.indices.shape)
G = csr_matrix((data,solver.indices,solver.inclusive),shape=(solver.N,solver.N))
L = csgraph.laplacian(G)
vals, _ = eigs(L, k=1)
M = vals[0]

if SOLVER_NAME == 'bdem':
    scene.cfl = 0.2
    solver = SolverBdem2(scene, neighborSearch)
elif SOLVER_NAME == 'mass_spring':
    scene.cfl = 0.2
    solver = SolverMassSpring2(scene, neighborSearch)
elif SOLVER_NAME == 'peridynamics':
    scene.cfl = 0.2
    scene.kn /= 100.0
    solver = SolverPeridynamics2(scene, neighborSearch)
elif SOLVER_NAME == 'mpm':
    scene.cfl = 1.0
    M = 1
    solver = SolverMpm2(scene, neighborSearch)
elif SOLVER_NAME == 'dem':
    solver = SolverDem2(scene, neighborSearch)
    
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
    npColor = solver.color.to_numpy()
    colors = np.zeros(npPosition[npLabel == 0].shape[0], dtype=np.uint32)
    for i, color in enumerate(npColor[npLabel == 0]):
        # c = np.uint32(color * 255.0)
        c = np.uint32(np.array(plt.cm.jet(color[0])) * 255.0)
        colors[i] = (c[0] << 16) | (c[1] << 8) | c[2]
    print('color max : ', hex(np.max(colors)))
    radius = WINDOW_SIZE * scene.r
    gui.circles(npPosition[npLabel == 0], radius=radius, color=colors) # 0088ff
    gui.circles(npPosition[npLabel != 0], radius=radius, color=0x000000)
    # # draw bonds
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