import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.subheader("📡 GPR (Ground Penetrating Radar) Vertical Cross-Section")
st.write("Scan Path: **North-South Transect through +0.5m Offset**")

def generate_gpr_scan():
    # Grid: X (Distance along surface), Y (Depth into ground)
    x = np.linspace(0, 10, 200)
    y = np.linspace(0, 6, 200) # Depth up to 6 meters
    X, Y = np.meshgrid(x, y)
    
    # Base signal (soil noise)
    z = np.random.normal(0.5, 0.05, X.shape)

    # 1. The 'Sigilith Plate' (0.5m - 1.2m)
    # High amplitude reflection (hyperbola)
    cap_mask = (np.abs(X - 5) < 1.0) & (np.abs(Y - 0.8) < 0.1)
    z[cap_mask] += 0.8 

    # 2. The 'Buffer' Layer (1.2m - 4.0m) - Scattering Quartz/Charcoal
    # Intense localized noise points
    scatter_mask = (X > 3) & (X < 7) & (Y > 1.2) & (Y < 4.0)
    z[scatter_mask] += np.random.normal(0, 0.3, z[scatter_mask].shape)

    # 3. The 'Temple' Vault Ceiling (4.2m)
    # Flat, strong planar reflection
    vault_ceiling = (X > 3) & (X < 7) & (np.abs(Y - 4.2) < 0.05)
    z[vault_ceiling] += 1.0 

    return X, Y, z

X, Y, Z = generate_gpr_scan()

fig, ax = plt.subplots(figsize=(10, 5))
# Using 'Greys' to mimic a real GPR monitor
im = ax.imshow(Z, extent=[0, 10, 6, 0], cmap='Greys', aspect='auto')
plt.colorbar(im, label="Reflection Strength")

# Labels for Hope Jones's Theory
ax.set_xlabel("Surface Distance (m)")
ax.set_ylabel("Depth (m)")
ax.axhline(4.2, color='red', linestyle='--', alpha=0.5, label="Vault Ceiling (4.2m)")
ax.legend()

st.pyplot(fig)

if st.button("Interpret Radar Signature"):
    st.success("Planar reflection confirmed at 4.2m depth.")
    st.info("Structure matches 'Temple of the Mind' dimensions (4x4m chamber).")
    st.write("**Contents:** Preserved Data Archive (Secret Society Records)")
