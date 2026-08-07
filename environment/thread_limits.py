# Pins BLAS/OpenMP to a single thread per process.
# Must be called before numpy is imported - the BLAS backends read these vars
# once at load time, so setting them afterwards has no effect.
# This module imports nothing but os, so it is safe to import first.
import os

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def limit_threads(n=1, torch_threads=None):
    """Pin BLAS/OpenMP to n threads per process."""
    value = str(n)
    for var in _THREAD_VARS:
        os.environ[var] = value

    # torch is optional here; it manages its own pool and ignores the env vars
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(n if torch_threads is None else torch_threads)

    return {var: os.environ[var] for var in _THREAD_VARS}
