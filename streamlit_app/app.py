import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from src.arp.io import load_candidates_geojson
from src.arp.fusion import compute_S
import matplotlib.pyplot as plt

st.set_page_config(page_title="Archaeo-Resonance Explorer", layout="wide")
st.title("Archaeo-Resonance Explorer")

# Sidebar controls
with st.sidebar:
    st.header("Fusion Controls")
    w_g = st.slider("w_g (geometry)", 0.0, 1.0, 0.18, 0.01)
    w_h = st.slider("w_h (harmonics)", 0.0, 1.0, 0.26, 0.01)
    w_m = st.slider("w_m (magnetic)", 0.0, 1.0, 0.18, 0.01)
    w_l = st.slider("w_l (LIDAR)", 0.0, 1.0, 0.22, 0.01)
    w_s = st.slider("w_s (symbolic)", 0.0, 1.0, 0.16, 0.01)
    theta = st.slider("theta (bias)", 0.0, 1.0, 0.5, 0.01)
    lam = st.slider("lambda (sigmoid)", 0.1, 10.0, 6.0, 0.1)

# Load candidates
candidates = load_candidates_geojson("examples/known_sites.geojson")

# Compute S(x) for each candidate dynamically
params = {"w": [w_g, w_h, w_m, w_l, w_s], "theta": theta, "lam": lam}
scores = []
feature_contribs = {"G": [], "H": [], "M": [], "L": [], "Ssym": []}

for _, row in candidates.iterrows():
    features = {
        "G": row.get("G", 0.5),
        "H": row.get("H", 0.5),
        "M": row.get("M", 0.5),
        "L": row.get("L", 0.5),
        "Ssym": row.get("Ssym", 0.5),
    }
    scores.append(compute_S(features, params))
    for k in features:
        feature_contribs[k].append(features[k] * params["w"][list(features.keys()).index(k)])

candidates["S"] = scores

# Show candidate table
st.write("## Candidate Sites")
st.dataframe(candidates)

# Feature contribution plot
st.write("## Feature Contributions")
contrib_df = pd.DataFrame(feature_contribs)
st.bar_chart(contrib_df)

# Create folium map
m = folium.Map(location=[candidates['lat'].mean(), candidates['lon'].mean()], zoom_start=6)
for _, row in candidates.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5 + row['S']*10,
        color='blue',
        fill=True,
        fill_opacity=0.6,
        popup=f"Score: {row['S']:.2f}"
    ).add_to(m)

st.write("## Candidate Map")
st_folium(m, width=800, height=600)

# Download GeoJSON
if st.button("Download Candidates GeoJSON"):
    candidates.to_file("candidates.geojson", driver="GeoJSON")
    st.success("GeoJSON saved as candidates.geojson")
