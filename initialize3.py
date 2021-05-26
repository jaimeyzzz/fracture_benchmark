import numpy as np
import sys
import taichi as ti

from neighbor_search3 import NeighborSearch3
from scene import Scene

ti.init(arch=ti.gpu)

SCENE_FOLDER = 'scene3'
SCENE_NAME = sys.argv[1]
SOLVER_NAME = sys.argv[2]

# INIT SCENE
scene = Scene(SCENE_FOLDER, SCENE_NAME)

if SOLVER_NAME == 'bdem':
    scene.h = scene.r * 3.01
elif SOLVER_NAME == 'dem':
    pass
elif SOLVER_NAME == 'mass_spring':
    scene.h = scene.r * 3.01
elif SOLVER_NAME == 'peridynamics':
    scene.h = scene.r * 6.01

N = scene.N
bondsNum = ti.field(ti.i32)
bondsIdx = ti.field(ti.i32)
label = ti.field(ti.f32)
position = ti.Vector.field(3, ti.f32)
ti.root.dense(ti.i, N).place(bondsNum)
ti.root.dense(ti.i, N).dense(ti.j, scene.MAX_BONDS_NUM).place(bondsIdx)
ti.root.dense(ti.i, N).place(label)
ti.root.dense(ti.i, N).place(position)
bondSearch = NeighborSearch3(scene.lowerBound, scene.upperBound, scene.h)
bondSearch.init()
label.from_numpy(scene.label)
position.from_numpy(np.delete(scene.position, -1, axis=1))
bondSearch.updateCells(position)

# COMPUTE BONDS
@ti.kernel
def computeBonds():
    for i in position:
        li = label[i]
        if li == scene.FLUID:
            pi = position[i]
            cell = bondSearch.getCell(pi)
            cnt = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                neighborCell = cell + offs
                if bondSearch.isCellInRange(neighborCell):
                    for k in range(bondSearch.cellsNum[neighborCell]):
                        j = bondSearch.cells[neighborCell, k]
                        lj = label[j]
                        # if lj != scene.FLUID: continue
                        pj = position[j]
                        if i != j and (pi - pj).norm() < scene.h:
                            bondsIdx[i, cnt] = j
                            cnt += 1
            bondsNum[i] = cnt

computeBonds()

# DUMP BONDS
filepath = '{}/{}/{}.npz'.format(SCENE_FOLDER, SCENE_NAME, SCENE_NAME)
npBondsNum = bondsNum.to_numpy()
npBondsIdx = bondsIdx.to_numpy()
inclusive = np.zeros(N + 1, dtype=int)
M = 0
for i in range(N):
    M = M + npBondsNum[i]
    inclusive[i + 1] = inclusive[i] + npBondsNum[i]
indices = np.zeros(M)
for i in range(N):
    for j in range(inclusive[i], inclusive[i + 1]):
        idx = j - inclusive[i]
        indices[j] = npBondsIdx[i][idx]
np.savez_compressed(filepath, inclusive=inclusive, indices=indices)

# EIGENS
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigs

data = np.ones(indices.shape)
G = csr_matrix((data,indices,inclusive),shape=(N,N))
L = csgraph.laplacian(G)

vals, vecs = eigs(L, k=6)
print(vals)