import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import plotly.express as px
import os
from shapely.geometry import Point
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Archaeo-Resonance Explorer", layout="wide")

# ---------------------------
# Utility: Load GeoJSON or CSV
# ---------------------------
@st.cache_data(show_spinner=False)
def load_candidates(path_or_file):
    """Load candidate sites from GeoJSON, JSON, or CSV."""
    if path_or_file is None:
        return None
    try:
        if hasattr(path_or_file, "read"):
            name = getattr(path_or_file, "name", "uploaded")
            if name.endswith(".geojson") or name.endswith(".json"):
                data = json.load(path_or_file)
                gdf = gpd.GeoDataFrame.from_features(data["features"])
                gdf.crs = "EPSG:4326"
                return gdf
            elif name.endswith(".csv"):
                return pd.read_csv(path_or_file)
        else:
            if str(path_or_file).endswith((".geojson", ".json")):
                gdf = gpd.read_file(path_or_file)
                gdf.set_crs(epsg=4326, inplace=True)
                return gdf
            elif str(path_or_file).endswith(".csv"):
                return pd.read_csv(path_or_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# ---------------------------
# Fusion score computation
# ---------------------------
def compute_S(G, H, M, L, Ssym, w_g, w_h, w_m, w_l, w_s, theta, lam):
    Lscore = w_g * G + w_h * H + w_m * M + w_l * L + w_s * Ssym
    return 1 / (1 + np.exp(-lam * (Lscore - theta)))


# ---------------------------
# Sidebar controls
# ---------------------------
st.title("🌍 Archaeo-Resonance Explorer")

st.sidebar.header("Fusion Controls")
w_g = st.sidebar.slider("w_g (geometry)", 0.0, 1.0, 0.18)
w_h = st.sidebar.slider("w_h (harmonics)", 0.0, 1.0, 0.26)
w_m = st.sidebar.slider("w_m (magnetic)", 0.0, 1.0, 0.18)
w_l = st.sidebar.slider("w_l (LIDAR)", 0.0, 1.0, 0.22)
w_s = st.sidebar.slider("w_s (symbolic)", 0.0, 1.0, 0.16)
theta = st.sidebar.slider("θ (bias)", 0.0, 1.0, 0.5)
lam = st.sidebar.slider("λ (sigmoid)", 0.1, 10.0, 6.0)

uploaded = st.file_uploader("Upload candidate sites (.geojson or .csv)")
example_options = {
    "Example A (GeoJSON)": "examples/known_sites_A.geojson",
    "Example B (GeoJSON)": "examples/known_sites_B.geojson",
}
selected_example = st.selectbox("Or choose an example", ["None"] + list(example_options.keys()))

# maintain persistent state
if "candidates" not in st.session_state:
    st.session_state["candidates"] = None

if uploaded:
    st.session_state["candidates"] = load_candidates(uploaded)
elif selected_example != "None":
    st.session_state["candidates"] = load_candidates(example_options[selected_example])

# Map mode moved OUTSIDE tabs to prevent reset
st.session_state["map_mode"] = st.radio(
    "🗺️ Choose map display mode:",
    ["Scatter Map", "Heatmap", "3D Terrain"],
    horizontal=True,
    index=["Scatter Map", "Heatmap", "3D Terrain"].index(st.session_state.get("map_mode", "Scatter Map")),
)

tabs = st.tabs(["Map", "Analytics", "About"])

candidates = st.session_state["candidates"]

# ---------------------------
# Main Logic
# ---------------------------
if candidates is not None:
    required_cols = ["G", "H", "M", "L", "Ssym"]
    missing = [c for c in required_cols if c not in candidates.columns]

    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        if st.button("🧩 Auto-fix missing columns"):
            for col in missing:
                candidates[col] = np.random.uniform(0.3, 0.8, len(candidates))
            st.session_state["candidates"] = candidates
            st.success("Added missing columns with random values.")
            missing = []

    if not missing:
        candidates["S"] = compute_S(
            candidates["G"], candidates["H"], candidates["M"],
            candidates["L"], candidates["Ssym"],
            w_g, w_h, w_m, w_l, w_s, theta, lam,
        )

        # make sure lat/lon exist
        if "geometry" in candidates:
            gdf = gpd.GeoDataFrame(candidates)
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            gdf["geometry"] = gdf.geometry.centroid
            gdf["lat"] = gdf.geometry.y
            gdf["lon"] = gdf.geometry.x
            candidates = gdf
        elif {"lat", "lon"}.issubset(candidates.columns):
            candidates["lat"] = candidates["lat"].astype(float)
            candidates["lon"] = candidates["lon"].astype(float)
        else:
            st.warning("No geometry or lat/lon columns found — cannot plot map.")

        # drop missing coords
        candidates = candidates.dropna(subset=["lat", "lon", "S"])

        # ---------------- MAP TAB ----------------
        with tabs[0]:
            st.subheader("Interactive Map Visualization")

            try:
                if st.session_state["map_mode"] == "Scatter Map":
                    fig = px.scatter_mapbox(
                        candidates,
                        lat="lat", lon="lon",
                        color="S", size="S",
                        color_continuous_scale="Viridis",
                        zoom=5, mapbox_style="open-street-map",
                        title="Site Likelihood Scatter Map",
                    )

                elif st.session_state["map_mode"] == "Heatmap":
                    fig = px.density_mapbox(
                        candidates,
                        lat="lat", lon="lon", z="S",
                        radius=20,
                        center=dict(lat=candidates["lat"].mean(), lon=candidates["lon"].mean()),
                        zoom=5, mapbox_style="open-street-map",
                        color_continuous_scale="YlOrRd",
                        title="Site Likelihood Heatmap",
                    )

                else:  # 3D Terrain
                    fig = px.scatter_3d(
                        candidates,
                        x="lon", y="lat", z="S",
                        color="S", color_continuous_scale="Viridis",
                        title="3D Terrain Resonance Map",
                    )
                    fig.update_traces(marker=dict(size=5, opacity=0.8))
                    fig.update_layout(scene=dict(
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        zaxis_title="Likelihood (S)",
                        aspectmode="cube",
                    ))

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not plot map: {e}")

        # ---------------- ANALYTICS TAB ----------------
        with tabs[1]:
            st.subheader("📊 Statistical Overview")
            st.dataframe(candidates[required_cols + ["S"]])
            st.plotly_chart(px.histogram(candidates, x="S", nbins=20, color_discrete_sequence=["#4B9CD3"]), use_container_width=True)
            st.plotly_chart(px.scatter_matrix(candidates, dimensions=required_cols + ["S"], color="S"), use_container_width=True)

        # ---------------- ABOUT TAB ----------------
        with tabs[2]:
            st.markdown("""
            ### ℹ️ About Archaeo-Resonance Explorer
            Combines geometry, harmonics, magnetic, LIDAR, and symbolic signals into a single probabilistic score:
            \[
            S = \frac{1}{1 + e^{-\lambda (w_g G + w_h H + w_m M + w_l L + w_s S_{sym} - \theta)}}
            \]
            **λ** controls sigmoid steepness, **θ** sets bias, and each **w\_*** weights a modality.
            """)
else:
    st.info("📂 Upload a file or choose an example to begin.")
