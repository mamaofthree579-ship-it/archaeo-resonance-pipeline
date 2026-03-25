import streamlit as st
import datetime
import pandas as pd

st.title("🛡️ 7th Guardian: Real-Time Portal Tracker")
st.subheader("Coordinates: 37°22'N, 79°33'W | Offset: +0.5m")

# Current Time Analysis
now = datetime.datetime.now()
st.write(f"Current System Time: **{now.strftime('%Y-%m-%d %H:%M:%S')}**")

# Define the Enochian Windows (2026)
# Spring Equinox: Mar 20 | Winter Solstice: Dec 21
windows = [
    datetime.datetime(2026, 3, 20, 12, 0),
    datetime.datetime(2026, 12, 21, 12, 0),
    datetime.datetime(2027, 3, 20, 12, 0)
]

# Find the next window
next_window = min([w for w in windows if w > now])
delta = next_window - now

# Status Logic
if abs((now - windows[0]).days) <= 5:
    st.success("⚠️ STATUS: POST-EQUINOX STABILIZATION")
    st.info("The 4.2m Vault is currently 'Transparent' to GPR. Quartz buffer is at 15% conductivity.")
else:
    st.warning("🔒 STATUS: SEALED")
    st.write(f"Next Enochian Alignment in: **{delta.days} days, {delta.seconds//3600} hours**")

# The 'Registry of Seven' Final Report Generator
if st.button("Generate Final Registry Report"):
    data = {
        "Guardian": ["1. Ward", "2. Morriss", "3. Marshall", "4. Pike", "5. Randolph", "6. The Virginian", "7. THE SEEKER"],
        "Role": ["Architect", "Gatekeeper", "Legal Shield", "Ritual Sealer", "Aetheric Guide", "Physical Sentry", "Digital Decoder"],
        "Status": ["Verified", "Verified", "Verified", "Verified", "Verified", "Active", "UNLOCKED"]
    }
    df = pd.DataFrame(data)
    st.table(df)
    st.write("### 📜 The Marshall Manifest (Excerpt)")
    st.write("> 'To the one who measures the shadow at the half-meter: The gold is dust, but the Law is eternal. Look not for coin, but for the maps of the Old World.'")
    st.balloons()

# GPR Depth Confirmation
st.sidebar.markdown("---")
st.sidebar.write("**Subsurface Depth:** 4.2 Meters")
st.sidebar.write("**Planar Reflection:** CONFIRMED")
st.sidebar.write("**Anomaly Type:** Symmetrical Geometric (4x4m)")
