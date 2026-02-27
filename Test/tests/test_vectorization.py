import numpy as np
import time
import pytest
from quantum_fold.core.force_field import (
    CoarseGrainedForceField,
    dfire2_potential,
    lennard_jones_energy,
    electrostatic_energy,
    solvation_energy,
)

@pytest.fixture
def sample_data():
    seq = "YYDPETGTWY"
    n = len(seq)
    coords = np.random.rand(n, 3) * 10.0
    return seq, coords

def test_dfire_vectorization(sample_data):
    seq, coords = sample_data
    e, grad = dfire2_potential(coords, seq)
    assert isinstance(e, float)
    assert grad.shape == coords.shape
    assert not np.isnan(e)
    assert not np.any(np.isnan(grad))

def test_lj_vectorization(sample_data):
    seq, coords = sample_data
    e, grad = lennard_jones_energy(coords, seq)
    assert isinstance(e, float)
    assert grad.shape == coords.shape
    assert not np.isnan(e)
    assert not np.any(np.isnan(grad))

def test_elec_vectorization(sample_data):
    seq, coords = sample_data
    e, grad = electrostatic_energy(coords, seq)
    assert isinstance(e, float)
    assert grad.shape == coords.shape
    assert not np.isnan(e)
    assert not np.any(np.isnan(grad))

def test_solv_vectorization(sample_data):
    seq, coords = sample_data
    e, grad = solvation_energy(coords, seq)
    assert isinstance(e, float)
    assert grad.shape == coords.shape
    assert not np.isnan(e)
    assert not np.any(np.isnan(grad))

def test_full_grad(sample_data):
    seq, coords = sample_data
    ff = CoarseGrainedForceField()
    e, grad = ff.score(coords, seq, return_grad=True)
    assert isinstance(e, float)
    assert grad.shape == coords.shape
    assert not np.isnan(e)
    assert not np.any(np.isnan(grad))

def test_performance_gain():
    # Larger system to see the difference
    n = 50
    seq = "A" * n
    coords = np.random.rand(n, 3) * 20.0
    ff = CoarseGrainedForceField()

    start = time.time()
    for _ in range(10):
        ff.score(coords, seq, return_grad=True)
    dt = time.time() - start
    print(f"\nTime for 10 evaluations (N=50): {dt:.4f}s")
    # This should be very fast now
    assert dt < 0.5
