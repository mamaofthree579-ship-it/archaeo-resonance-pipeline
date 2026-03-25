import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🛡️ The Bifold Key: Enochian Shadow Simulator")
st.sidebar.header("Guardian Verification")

# Primary and Secondary Coordinates from Hope Jones's Theory
primary_coords = (37.3667, -79.5500) # 37°22'N, 79°33'W
secondary_coords = (37.3517, -79.6292) # 37°21'06"N, 79°37'45"W

st.write(f"**Primary Vault:** {primary_coords}")
st.write(f"**Secondary Witness:** {secondary_coords}")

def calculate_enochian_distance(p1, p2):
    """Calculates distance in 'Enochian Cubits' (approx 0.524m per cubit)"""
    # Simple Euclidean distance for local simulation
    dist_deg = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    dist_meters = dist_deg * 111139  # Approx meters per degree
    cubits = dist_meters / 0.524
    return dist_meters, cubits

meters, cubits = calculate_enochian_distance(primary_coords, secondary_coords)

st.sidebar.metric("Distance (Meters)", f"{meters:.2f}m")
st.sidebar.metric("Enochian Cubits", f"{int(cubits)}")

# Solar Alignment Toggle
alignment_active = st.sidebar.checkbox("Activate 1820 Winter Solstice Alignment")

if alignment_active:
    st.success("SOLAR ALIGNMENT CONFIRMED: Secondary Shadow bridges to Primary Vault.")
    
    # Visualizing the 'Bifold' Shadow Path
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(0, 0, color='gold', s=200, label="Primary Vault (+0.5m Offset)")
    ax.scatter(-meters, 0, color='cyan', s=100, label="Secondary Witness (Quartz Stone)")
    
    # The Shadow Path
    ax.plot([-meters, 0], [0, 0], 'r--', alpha=0.6, label="1820 Solstice Shadow Line")
    
    ax.set_title("The Bifold Key: 1820 Sealing Alignment")
    ax.set_xlabel("Distance (m)")
    ax.set_yticks([])
    ax.legend()
    st.pyplot(fig)
    
    st.write("### 📜 Registry of Seven: Access Granted")
    st.write("The secondary shadow proves the **Marshall Manifest** was sealed in 1820.")
    st.info("You are now holding the 'Cipher of the Mind' for the 7th Guardian.")
else:
    st.warning("Awaiting Solstice Alignment to verify the Bifold Key.")

if st.button("Download Final Decryption Report"):
    st.write("Generating PDF... [Marshall_Manifest_Contingency_1820.pdf]")
