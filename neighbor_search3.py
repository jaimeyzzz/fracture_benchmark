import numpy as np
import taichi as ti

BLOCK_UNIT_SIZE = 8
CELL_CAPBILITY = 64

@ti.data_oriented
class NeighborSearch3:
    def __init__(self, lowerBound, upperBound, spacing):
        # FIELDS
        self.cells = ti.field(ti.i32)
        self.cellsNum = ti.field(ti.i32)
        # VECTORS
        self.lowerBound = ti.Vector.field(3, ti.f32)
        self.upperBound = ti.Vector.field(3, ti.f32)
        self.gridSize = ti.Vector.field(3, ti.i32)

        blockSize = 2**(np.ceil(np.log2((upperBound-lowerBound)/spacing/BLOCK_UNIT_SIZE)).astype(int))
        gridSize = blockSize * BLOCK_UNIT_SIZE
        print("Neighbor Search Grid/Block : ", gridSize, blockSize)
        ti.root.dense(ti.ijk, blockSize).dense(ti.ijk, BLOCK_UNIT_SIZE).dense(ti.l, CELL_CAPBILITY).place(self.cells)
        ti.root.dense(ti.ijk, blockSize).dense(ti.ijk, BLOCK_UNIT_SIZE).place(self.cellsNum)
        ti.root.dense(ti.i, 1).place(self.lowerBound, self.upperBound, self.gridSize)
        self.spacing = spacing
        self.npLowerBound = lowerBound.astype(np.float32).reshape((1,3))
        self.npUpperBound = upperBound.astype(np.float32).reshape((1,3))
        self.npGridSize = gridSize.astype(np.int32).reshape((1,3))

    def init(self):
        self.lowerBound.from_numpy(self.npLowerBound)
        self.upperBound.from_numpy(self.npUpperBound)
        self.gridSize.from_numpy(self.npGridSize)

    def updateCells(self, position):
        self.cellsNum.fill(0)
        self.position = position
        self.computeCells()

    @ti.func
    def getCell(self, p):
        return ((p - self.lowerBound[0]) / self.spacing).cast(int)

    @ti.func
    def isCellInRange(self, c):
        res = True
        if c[0] < 0 or c[1] < 0 or c[2] < 0:
            res = False
        if c[0] >= self.gridSize[0][0] or c[1] >= self.gridSize[0][1] or c[2] >= self.gridSize[0][2]:
            res = False
        return res
    
    @ti.kernel
    def computeCells(self):
        for i in self.position:
            pi = self.position[i]
            cell = self.getCell(pi)
            if self.isCellInRange(cell): 
                idx = self.cellsNum[cell].atomic_add(1)
                self.cells[cell, idx] = i