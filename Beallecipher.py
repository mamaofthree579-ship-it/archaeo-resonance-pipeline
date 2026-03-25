import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import datetime

st.header("🌞 Enochian Solar Alignment Simulator")
st.write("Target Date: **Spring Equinox (March 21st)**")

# Theory Parameters
target_date = st.sidebar.date_input("Alignment Date", datetime.date(2026, 3, 21))
time_of_day = st.sidebar.slider("Time (Solar Noon focus)", 8, 16, 12)

def simulate_solar_shadow(hour):
    # Create the Vesica Piscis surface again
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # The Vesica Piscis Depression
    dist1 = np.sqrt((X + 4)**2 + Y**2)
    dist2 = np.sqrt((X - 4)**2 + Y**2)
    vesica = (dist1 < 8) & (dist2 < 8)
    Z = np.zeros_like(X)
    Z[vesica] = -0.2  # 20cm surface depression
    
    # Calculate shadow angle based on hour (12 = vertical/noon)
    # At Solar Noon on the Equinox at 37°N, the angle is specific
    shadow_offset = (hour - 12) * 2.0
    shadow_mask = (np.abs(X - shadow_offset) < 0.3) & (np.abs(Y) < 5)
    
    return X, Y, Z, shadow_mask

X, Y, Z, shadow = simulate_solar_shadow(time_of_day)

fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(X, Y, Z, cmap='copper', alpha=0.6) # Terrain
ax.contourf(X, Y, shadow, cmap='binary', alpha=0.9) # The Shadow "Key"

# Mark the +0.5m Anomaly
ax.scatter([0.5], [0], color='cyan', s=100, label="Sigilith Plate (+0.5m)")
ax.set_title(f"Solar Alignment at {time_of_day}:00")
ax.legend()

st.pyplot(fig)

# The "Activation" Logic
if time_of_day == 12 and target_date.month == 3 and target_date.day == 21:
    st.balloons()
    st.success("ALIGNMENT CONFIRMED: Shadow bisects the +0.5m Offset.")
    st.info("The 'Enochian Portal' is active. Quartz buffer conductivity is at minimum.")
    st.write("### 📜 Decrypted Documents Accessible:")
    st.write("- **The Marshall Manifest**: 1820 Constitutional Contingency")
    st.write("- **The Enochian Map**: Pre-Columbian Blue Ridge Settlements")
    st.write("- **The Registry of Seven**: Final lineage of the Guardians")
else:
    st.warning("Alignment Pending. The shadow is not yet centered on the Sigilith Plate.")
