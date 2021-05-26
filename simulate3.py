import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import taichi as ti
import time

from neighbor_search3 import NeighborSearch3
from scene import Scene
from solver_base3 import SolverBase3
from solver_bdem3 import SolverBdem3
from solver_dem3 import SolverDem3
from solver_mass_spring3 import SolverMassSpring3
from solver_mpm3 import SolverMpm3
from solver_peridynamics3 import SolverPeridynamics3

#ti.init(debug=True, log_level=ti.TRACE)
ti.init(arch=ti.gpu)

SCENE_FOLDER = 'scene3'
SCENE_NAME = sys.argv[1]
SOLVER_NAME = sys.argv[2]
NUM_FRAMES = int(sys.argv[3])
M = 1
WINDOW_SIZE = 640

scene = Scene(SCENE_FOLDER, SCENE_NAME)
outputDir = 'output/data/{}_{}3/'.format(SCENE_NAME, SOLVER_NAME)
try:
    os.mkdir(outputDir)
except OSError as error:
    print(error)   

neighborSearch = NeighborSearch3(scene.lowerBound, scene.upperBound, scene.r * 2.0)
solver = SolverBase3(scene, neighborSearch)
# EIGENS
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs, eigsh

data = np.ones(solver.indices.shape)
G = csr_matrix((data,solver.indices,solver.inclusive),shape=(solver.N,solver.N))
L = csgraph.laplacian(G)
vals, _ = eigsh(L, k=1)
M = vals[0]

if SOLVER_NAME == 'bdem':
    scene.cfl = 0.2
    solver = SolverBdem3(scene, neighborSearch)
elif SOLVER_NAME == 'dem':
    solver = SolverDem3(scene, neighborSearch)
elif SOLVER_NAME == 'mass_spring':
    solver = SolverMassSpring3(scene, neighborSearch)
elif SOLVER_NAME == 'peridynamics':
    scene.kn = 1e4
    scene.kt = 1e4
    solver = SolverPeridynamics3(scene, neighborSearch)
elif SOLVER_NAME == 'mpm':
    scene.cfl = 0.2
    M = 1
    solver = SolverMpm3(scene, neighborSearch)
    
def dump(solver, frameIdx):
    # solver.surface()
    # num_vertices = solver.vertices.shape[0]
    # num_faces = solver.faces.shape[0]
    # if num_vertices == 0: 
    #     print('[DEM] dump frame {}'.format(frameIdx))
    #     return
    # writer = ti.PLYWriter(num_vertices=num_vertices,num_faces=num_faces)
    # writer.add_vertex_pos(solver.vertices[:,0], solver.vertices[:,1], solver.vertices[:,2])
    # writer.add_vertex_rgba(solver.colors[:,0], solver.colors[:,1], solver.colors[:,2], solver.colors[:,3])
    # writer.add_faces(solver.faces)

    # writer.export_frame(frameIdx, 'output/surface')

    #solver.setColor()
    npPosition = solver.position.to_numpy()
    npRadiuses = solver.radius.to_numpy()
    npLabel = solver.label.to_numpy()
    npMass = solver.mass.to_numpy()
    
    # npColor = solver.color.to_numpy().reshape((-1,4))
    npColor = solver.color.to_numpy()
    colors = np.zeros((solver.N, 4), dtype=np.float32)
    for i, color in enumerate(npColor):
        colors[i, 3] = 1.0
        if npLabel[i] == solver.scene.FLUID:
            colors[i] = np.array(plt.cm.jet(color[0]))
    npV = solver.velocity.to_numpy()
    # npW = solver.angularVelocity.to_numpy()
    # ply writer
    writer = ti.PLYWriter(num_vertices=solver.N)
    writer.add_vertex_pos(npPosition[:,0], npPosition[:,1], npPosition[:,2])
    writer.add_vertex_channel("pscale", "double", npRadiuses.reshape((-1,1)))
    writer.add_vertex_channel("m", "double", npMass.reshape((-1,1)))
    writer.add_vertex_channel("vx", "double", npV[:,0])
    writer.add_vertex_channel("vy", "double", npV[:,1])
    writer.add_vertex_channel("vz", "double", npV[:,2])
    # writer.add_vertex_channel("wx", "double", npW[:,0])
    # writer.add_vertex_channel("wy", "double", npW[:,1])
    # writer.add_vertex_channel("wz", "double", npW[:,2])
    writer.add_vertex_rgba(colors[:,0],colors[:,1],colors[:,2],colors[:,3])
    #writer.add_faces(solver.triangles)
    writer.export_frame(frameIdx, outputDir+'frame')
    print('[DEM] dump frame {}'.format(frameIdx))

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
dump(solver, frameIdx)
print('Process frame', frameIdx)
while gui.running and frameIdx < NUM_FRAMES:
    start = time.time()
    for _ in range(s):
        solver.update(dt)

        load = solver.load.to_numpy()[0]
        npLoad = np.append(npLoad, np.linalg.norm(load))
        npTime = np.append(npTime, solver.t[0])
    end = time.time()
    print(load, solver.t[0])

    npPosition = solver.position.to_numpy()
    npLabel = solver.label.to_numpy()
    radius = WINDOW_SIZE * scene.r
    # gui.circles(npPosition[npLabel == 0], radius=radius, color=0xffaa11) # 0088ff
    # gui.circles(npPosition[npLabel != 0], radius=radius, color=0x000000)
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
    print('Process frame', frameIdx, 'for', end - start)
    dump(solver, frameIdx)
    # gui.show(outputDir+f'{frameIdx:04d}.jpg')

np.savez('output/{}_{}3.npz'.format(SCENE_NAME, SOLVER_NAME), load=npLoad, time=npTime)