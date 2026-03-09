from __future__ import annotations

import os


# Keep BLAS thread fan-out bounded under pytest-xdist workers.
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")


def pytest_xdist_auto_num_workers(config) -> int:
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, 8)
