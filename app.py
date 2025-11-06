import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import plotly.express as px
import os
from shapely.geometry import Point
from PIL import Image

# ---------------------------
# Utility: Load GeoJSON or CSV
# ---------------------------
def load_candidates(path_or_file):
    """Load candidate sites from GeoJSON, JSON, or CSV."""
    if path_or_file is None:
        return None

    try:
        # Handle uploaded file-like object or file path
        if hasattr(path_or_file, "read"):
            name = getattr(path_or_file, "name", "uploaded")
            if name.endswith(".geojson") or name.endswith(".json"):
                data = json.load(path_or_file)
                gdf = gpd.GeoDataFrame.from_features(data["features"])
                gdf.crs = "EPSG:4326"  # default WGS84
                return gdf
            elif name.endswith(".csv"):
                return pd.read_csv(path_or_file)
            else:
                st.error("Unsupported file format. Please upload .geojson or .csv.")
                return None
        else:
            if str(path_or_file).endswith((".geojson", ".json")):
                gdf = gpd.read_file(path_or_file)
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                return gdf
            elif str(path_or_file).endswith(".csv"):
                return pd.read_csv(path_or_file)
            else:
                st.error("Unsupported example file format.")
                return None
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
# Streamlit UI
# ---------------------------
st.title("🌍 Archaeo-Resonance Explorer")

st.markdown(
    "Upload your candidate site data (`.geojson` or `.csv`) or choose a built-in example to compute site-likelihood scores."
)

# Fusion control sliders
st.sidebar.header("Fusion Controls")
w_g = st.sidebar.slider("w_g (geometry)", 0.0, 1.0, 0.18)
w_h = st.sidebar.slider("w_h (harmonics)", 0.0, 1.0, 0.26)
w_m = st.sidebar.slider("w_m (magnetic)", 0.0, 1.0, 0.18)
w_l = st.sidebar.slider("w_l (LIDAR)", 0.0, 1.0, 0.22)
w_s = st.sidebar.slider("w_s (symbolic)", 0.0, 1.0, 0.16)
theta = st.sidebar.slider("θ (bias)", 0.0, 1.0, 0.5)
lam = st.sidebar.slider("λ (sigmoid)", 0.1, 10.0, 6.0)

# Upload or select example
uploaded = st.file_uploader("Upload candidate sites (.geojson or .csv)")
example_options = {
    "Example A (GeoJSON)": "examples/known_sites_A.geojson",
    "Example B (GeoJSON)": "examples/known_sites_B.geojson",
}
selected_example = st.selectbox("Or choose an example", ["None"] + list(example_options.keys()))

# Determine input source
if uploaded:
    candidates = load_candidates(uploaded)
    image_path = None
elif selected_example != "None":
    candidates = load_candidates(example_options[selected_example])
    image_path = os.path.splitext(example_options[selected_example])[0] + ".png"
else:
    candidates = None
    image_path = None

# ---------------------------
# Compute & visualize
# ---------------------------
if candidates is not None:
    required_cols = ["G", "H", "M", "L", "Ssym"]
    missing = [c for c in required_cols if c not in candidates.columns]

    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")

        if st.button("🧩 Auto-fix missing columns"):
            for col in missing:
                candidates[col] = np.random.uniform(0.3, 0.8, size=len(candidates))
            st.success("Added missing columns with default random values.")
            missing = []

    if not missing:
        candidates["S"] = compute_S(
            candidates["G"],
            candidates["H"],
            candidates["M"],
            candidates["L"],
            candidates["Ssym"],
            w_g,
            w_h,
            w_m,
            w_l,
            w_s,
            theta,
            lam,
        )

        st.success(f"✅ Computed site likelihoods for {len(candidates)} candidates.")
        st.dataframe(candidates[required_cols + ["S"]])

        # Map visualization (handle all geometry types)
        if "geometry" in candidates:
            try:
                gdf = gpd.GeoDataFrame(candidates)
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)

                # If not points, use centroids
                if not all(gdf.geometry.geom_type == "Point"):
                    gdf["geometry"] = gdf.geometry.centroid

                # Extract coordinates for plotting
                gdf["lat"] = gdf.geometry.y
                gdf["lon"] = gdf.geometry.x

                fig = px.scatter_mapbox(
                    gdf,
                    lat="lat",
                    lon="lon",
                    color="S",
                    color_continuous_scale="Viridis",
                    size="S",
                    zoom=6,
                    title="🗺️ Site Likelihood Map",
                    mapbox_style="open-street-map",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot map: {e}")
        else:
            st.info("No geometry found — showing only table view.")

        # Optional PNG overlay preview
        if image_path and os.path.exists(image_path):
            st.image(Image.open(image_path), caption="Associated Site Map", use_container_width=True)
else:
    st.info("📂 Upload a file or choose an example to begin.")
