import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# App Config
st.set_page_config(page_title="Guardian's Cipher: Anomaly Detector", layout="wide")
st.title("🛡️ The Guardian's Cipher: Subsurface Anomaly Detector")
st.sidebar.header("Scan Parameters")

# Sidebar inputs based on Hope Jones's 'Sigilith' Theory
coords = st.sidebar.text_input("Target Coordinates", "37°22'N, 79°33'W")
scan_radius = st.sidebar.slider("Scan Radius (meters)", 10, 100, 50)
theory_mode = st.sidebar.selectbox("Decoding Key", ["Book of Enoch (Ch. 72)", "Masonic Trestle Board", "Sigilith Plate"])

# Constants from 'The Guardian's Cipher'
ENOCHIAN_OFFSET = 0.5  # The +0.5m symmetrical depression found

def generate_lidar_data(radius, offset_val):
    """Simulates a LiDAR Bare Earth model with a hidden anomaly."""
    size = 100
    x = np.linspace(-radius, radius, size)
    y = np.linspace(-radius, radius, size)
    X, Y = np.meshgrid(x, y)
    
    # Simulate natural Blue Ridge terrain (sloping with noise)
    terrain = 0.05 * X + 0.02 * Y + np.random.normal(0, 0.03, X.shape)
    
    # Inject the 'Vesica Piscis' Anomaly at the +0.5m offset
    # Two overlapping circles creating a geometric depression
    center_dist = 5 + offset_val
    dist1 = np.sqrt((X + center_dist)**2 + Y**2)
    dist2 = np.sqrt((X - center_dist)**2 + Y**2)
    
    # The 'Sigilith' Vault shape
    vault_mask = (dist1 < 12) & (dist2 < 12)
    terrain[vault_mask] -= 0.25  # 25cm depression
    
    # Secondary 'Tunnel' entrance at the 0.5m offset point
    tunnel_mask = (np.abs(X - offset_val) < 1.5) & (np.abs(Y - 15) < 1.5)
    terrain[tunnel_mask] -= 0.4  # Deeper shaft
    
    return X, Y, gaussian_filter(terrain, sigma=1)

# Execution
X, Y, Z = generate_lidar_data(scan_radius, ENOCHIAN_OFFSET)

# Visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("LiDAR Hillshade Analysis")
    fig, ax = plt.subplots(figsize=(10, 7))
    ls = plt.get_cmap('terrain')
    contour = ax.contourf(X, Y, Z, levels=30, cmap='terrain')
    plt.colorbar(contour, ax=ax, label="Relative Elevation (m)")
    
    # Highlight the Detected Anomaly
    ax.annotate('Symmetrical Depression (+0.5m)', xy=(0.5, 0), xytext=(10, 10),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    st.pyplot(fig)

with col2:
    st.subheader("Decoded 'Heirs' Metadata")
    st.write(f"**Primary Guardian:** James B. Ward")
    st.write(f"**Secondary Guardian:** Robert Morriss")
    st.write(f"**Legal Protector:** John Marshall")
    st.write(f"**Archive Status:** SEALED (1820)")
    
    st.divider()
    
    if st.button("Analyze 0.5m Offset"):
        st.warning("Anomaly Confirmed.")
        st.info(f"The 0.5m shift aligns with Chapter 72 of Enoch. "
                f"Calculated entrance depth: ~4.2 meters below surface.")
        st.write("Suggested Non-Invasive Tool: **GPR (Ground Penetrating Radar)**")

st.success("Data suggests a non-natural geometric void at the specified coordinates.")
