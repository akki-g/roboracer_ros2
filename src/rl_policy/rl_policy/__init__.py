"""RL policy for the car: NumPy actor, observation builder, safety layer, ROS node."""

import os

# One BLAS thread. The policy's matrices are small: waking a BLAS thread pool (OpenBLAS
# starts one thread per core) for a CNN layer costs more than the math and gives
# multi-millisecond latency spikes on the Jetson. This runs before any module here
# imports NumPy; a value already set in the environment wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
