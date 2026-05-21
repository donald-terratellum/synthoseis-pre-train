import math

from train import _compute_qc_overhead_ratio, _is_qc_overhead_within_budget


def test_compute_qc_overhead_ratio_nominal():
    ratio = _compute_qc_overhead_ratio(qc_time_sec=2.0, total_batch_time_sec=10.0)
    assert math.isclose(ratio, 0.2)


def test_compute_qc_overhead_ratio_guards_zero_total_time():
    ratio = _compute_qc_overhead_ratio(qc_time_sec=1.0, total_batch_time_sec=0.0)
    assert ratio == 0.0


def test_qc_overhead_budget_check():
    assert _is_qc_overhead_within_budget(overhead_ratio=0.25, max_ratio=0.25)
    assert _is_qc_overhead_within_budget(overhead_ratio=0.2, max_ratio=0.25)
    assert not _is_qc_overhead_within_budget(overhead_ratio=0.3, max_ratio=0.25)
