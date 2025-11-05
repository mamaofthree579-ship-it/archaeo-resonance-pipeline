"""
Fusion module for computing final site-likelihood score S(x)
"""

def compute_S(features: dict, params: dict) -> float:
    """
    Compute site likelihood score S(x) from normalized features and params.

    Args:
        features (dict): normalized features with keys 'G','H','M','L','Ssym'
        params (dict): parameters with keys 'w' (weights list), 'theta', 'lam'

    Returns:
        float: site-likelihood score S (0..1)
    """
    import math

    # Extract weights and parameters
    w = params.get("w", [0.18, 0.26, 0.18, 0.22, 0.16])
    theta = params.get("theta", 0.5)
    lam = params.get("lam", 6)

    # Ensure feature keys exist
    required_keys = ["G", "H", "M", "L", "Ssym"]
    for k in required_keys:
        if k not in features:
            raise ValueError(f"Missing feature key: {k}")

    comps = [features[k] for k in required_keys]

    # Weighted sum
    L = sum(w_i * c_i for w_i, c_i in zip(w, comps))

    # Sigmoid transformation
    S = 1 / (1 + math.exp(-lam * (L - theta)))

    return S
  
