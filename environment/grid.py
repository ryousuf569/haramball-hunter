# Pitch-control grid geometry. Its own module because termination.py needs the
# cell size to bin a position, and lowblock_env imports termination -- keeping
# these here is what stops that from being a circular import.

PC_NX, PC_NY = 31, 34
PC_CELL_SIZE = 2.0
