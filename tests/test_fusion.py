import pytest
from src.arp.fusion import compute_S


def test_compute_S_basic():
    features = {"G": 0.75, "H": 0.64, "M": 0.70, "L": 0.58, "Ssym": 0.35}
    params = {"w": [0.18, 0.26, 0.18, 0.22, 0.16], "theta": 0.5, "lam": 6}
    score = compute_S(features, params)
    assert 0 <= score <= 1
