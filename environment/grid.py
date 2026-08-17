# Pitch-control grid geometry. Its own module because termination.py needs the
# cell size to bin a position, and the env imports termination -- keeping these
# here is what stops that from being a circular import.

import numpy as np

from physics.engine import local_to_global

PC_NX, PC_NY = 31, 34
PC_CELL_SIZE = 2.0

# Global-frame bounds of the grid: 62 x 68 m starting behind the halfway circle,
# i.e. x in [43, 105]. Ordered for imshow's extent=(left, right, bottom, top).
PC_EXTENT = (43.0, 43.0 + PC_NX * PC_CELL_SIZE, 0.0, PC_NY * PC_CELL_SIZE)


def make_ppcf_grid():
    # (n_cells, 2) cell centres in the global frame, matching the (x, y) index
    # order the callers reshape back to (PC_NX, PC_NY).
    x = (np.arange(PC_NX) + 0.5) * PC_CELL_SIZE
    y = (np.arange(PC_NY) + 0.5) * PC_CELL_SIZE
    X, Y = np.meshgrid(x, y, indexing="ij")
    return local_to_global(np.stack([X.ravel(), Y.ravel()], axis=1))


def cell_of(position):
    # Grid indices for a global-frame position, or None if it falls outside.
    i = int((float(position[0]) - PC_EXTENT[0]) // PC_CELL_SIZE)
    j = int(float(position[1]) // PC_CELL_SIZE)
    if 0 <= i < PC_NX and 0 <= j < PC_NY:
        return i, j
    return None
