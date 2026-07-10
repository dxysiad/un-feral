import pytest

from feral.utils import (
    MAX_AUTO_NUM_WORKERS,
    resolve_num_workers,
    suggested_num_workers,
)


# ===================================================================
# resolve_num_workers
# ===================================================================

class TestResolveNumWorkers:
    def test_explicit_int_passthrough(self):
        # An explicit non-negative int is honored verbatim (users can pin it).
        assert resolve_num_workers(0) == 0
        assert resolve_num_workers(4) == 4
        assert resolve_num_workers(64) == 64

    def test_minus_one_is_auto_capped_int(self):
        n = resolve_num_workers(-1)
        assert isinstance(n, int)
        assert 1 <= n <= MAX_AUTO_NUM_WORKERS

    def test_auto_matches_min_cap_available(self):
        avail = suggested_num_workers() or 1
        assert resolve_num_workers(-1) == max(1, min(MAX_AUTO_NUM_WORKERS, avail))

    def test_auto_respects_cap(self):
        # A tiny cap always wins over the machine's CPU count.
        assert resolve_num_workers(-1, cap=2) <= 2
        assert resolve_num_workers(-1, cap=1) == 1

    def test_invalid_values_raise(self):
        for bad in ("auto", -2, 1.5, True, None):
            with pytest.raises(ValueError):
                resolve_num_workers(bad)
