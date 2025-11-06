import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import plotly.express as px
import requests
import rasterio

# ---------------------------
# Utility: Load candidates
# ---------------------------
def load_candidates(path_or_file):
    if path_or_file is None:
        return None
    try:
        if hasattr(path_or_file, "read"):
            name = getattr(path_or_file, "name", "uploaded")
            if name.endswith(".geojson") or name.endswith(".json"):
                data = json.load(path_or_file)
                gdf = gpd.GeoDataFrame.from_features(data["features"])
                return gdf
            elif name.endswith(".csv"):
                return pd.read_csv(path_or_file)
        else:
            if str(path_or_file).endswith(".geojson"):
                return gpd.read_file(path_or_file)
            elif str(path_or_file).endswith(".csv"):
                return pd.read_csv(path_or_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ---------------------------
# Fusion computation
# ---------------------------
def compute_S(G, H, M, L, Ssym, w_g, w_h, w_m, w_l, w_s, theta, lam):
    Lscore = w_g * G + w_h * H + w_m * M + w_l * L + w_s * Ssym
    S = 1 / (1 + np.exp(-lam * (Lscore - theta)))
    return S

# ---------------------------
# DEM handling
# ---------------------------
def get_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()["results"][0]["elevation"]
    except Exception:
        return np.nan
    return np.nan

def extract_elevation_from_raster(raster_path, lat, lon):
    try:
        with rasterio.open(raster_path) as src:
            for la, lo in zip(lat, lon):
                row, col = src.index(lo, la)
                yield src.read(1)[row, col]
    except Exception:
        yield np.nan

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Archaeo-Resonance Explorer", layout="wide")
st.title("🌍 Archaeo-Resonance Explorer")

st.markdown("""
Upload or select example candidate sites to compute and visualize resonance likelihoods.
Use the **Fusion Controls** in the sidebar to adjust weighting.
""")

# Sidebar controls
st.sidebar.header("Fusion Controls")
w_g = st.sidebar.slider("w_g (geometry)", 0.0, 1.0, 0.18)
w_h = st.sidebar.slider("w_h (harmonics)", 0.0, 1.0, 0.26)
w_m = st.sidebar.slider("w_m (magnetic)", 0.0, 1.0, 0.18)
w_l = st.sidebar.slider("w_l (LIDAR)", 0.0, 1.0, 0.22)
w_s = st.sidebar.slider("w_s (symbolic)", 0.0, 1.0, 0.16)
theta = st.sidebar.slider("θ (bias)", 0.0, 1.0, 0.5)
lam = st.sidebar.slider("λ (sigmoid)", 0.1, 10.0, 6.0)

# Upload or example selection
uploaded = st.file_uploader("Upload candidate sites (.geojson or .csv)")
example_options = {
    "Example A (GeoJSON)": "examples/known_sites_A.geojson",
    "Example B (GeoJSON)": "examples/known_sites_B.geojson",
}
selected_example = st.selectbox("Or choose an example", ["None"] + list(example_options.keys()))

# Optional DEM
dem_file = st.file_uploader("Optional DEM (.tif or .asc)", type=["tif", "asc"])

if uploaded:
    candidates = load_candidates(uploaded)
elif selected_example != "None":
    candidates = load_candidates(example_options[selected_example])
else:
    candidates = None

# ---------------------------
# Main logic
# ---------------------------
if candidates is not None:
    required_cols = ["G", "H", "M", "L", "Ssym"]
    missing = [c for c in required_cols if c not in candidates.columns]

    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")
        if st.button("🧩 Auto-fix missing columns"):
            for col in missing:
                candidates[col] = np.random.uniform(0.3, 0.8, len(candidates))
            st.success("Added missing columns with default random values.")
            missing = []

    if not missing:
        candidates["S"] = compute_S(
            candidates["G"], candidates["H"], candidates["M"],
            candidates["L"], candidates["Ssym"],
            w_g, w_h, w_m, w_l, w_s, theta, lam
        )

        # If no geometry, create fake lat/lon grid
        if "geometry" in candidates:
            try:
                gdf = gpd.GeoDataFrame(candidates, geometry="geometry", crs="EPSG:4326")
                candidates["lat"] = gdf.geometry.centroid.y
                candidates["lon"] = gdf.geometry.centroid.x
            except Exception:
                st.warning("Geometry column found but could not extract coordinates.")
        else:
            if "lat" not in candidates or "lon" not in candidates:
                st.warning("No spatial data found. Generating random coordinates.")
                candidates["lat"] = np.random.uniform(35, 40, len(candidates))
                candidates["lon"] = np.random.uniform(-5, 5, len(candidates))

        # Tabs for map views
        tabs = st.tabs(["📊 Scatter", "🔥 Heatmap", "🏔️ 3D Terrain"])

        # Scatter Map
        with tabs[0]:
            st.subheader("Scatter Map")
            fig = px.scatter_mapbox(
                candidates, lat="lat", lon="lon", color="S",
                color_continuous_scale="Viridis", zoom=5,
                hover_data=["S"], size="S", mapbox_style="open-street-map"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        with tabs[1]:
            st.subheader("Heatmap View")
            try:
                fig = px.density_mapbox(
                    candidates, lat="lat", lon="lon", z="S",
                    radius=20, zoom=5, mapbox_style="stamen-terrain",
                    color_continuous_scale="Plasma"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create heatmap: {e}")

        # 3D Terrain
        with tabs[2]:
            st.subheader("3D Terrain View")
            candidates["elevation"] = np.nan

            if dem_file is not None:
                st.info("Extracting elevation from uploaded DEM...")
                candidates["elevation"] = list(extract_elevation_from_raster(
                    dem_file, candidates["lat"], candidates["lon"]
                ))
            else:
                st.info("Fetching elevation automatically...")
                candidates["elevation"] = candidates.apply(
                    lambda r: get_elevation(r["lat"], r["lon"]), axis=1
                )

            candidates["z"] = candidates["elevation"].fillna(0) + candidates["S"] * 100
            try:
                fig = px.scatter_3d(
                    candidates, x="lon", y="lat", z="z",
                    color="S", color_continuous_scale="Viridis",
                    size="S", title="3D Terrain + Resonance Score"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render 3D terrain: {e}")

        # About tab
        st.markdown("""
        ### About
        This app combines multi-sensor archaeological fusion scoring with real-world terrain elevation.
        You can upload candidate sites (.geojson / .csv) and optional DEMs (.tif / .asc) to visualize resonance likelihoods.
        """)

else:
    st.info("👋 Upload a candidate sites file or select an example from the dropdown to get started.")
