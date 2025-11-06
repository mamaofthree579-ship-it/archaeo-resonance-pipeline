import streamlit as st
import pandas as pd
import geopandas as gpd
import json
import numpy as np
import plotly.express as px

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
                return gdf
            elif name.endswith(".csv"):
                return pd.read_csv(path_or_file)
            else:
                st.error("Unsupported file format. Please upload .geojson or .csv.")
                return None
        else:
            # Local file path
            if str(path_or_file).endswith(".geojson") or str(path_or_file).endswith(".json"):
                gdf = gpd.read_file(path_or_file)
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
    Lscore = (
        w_g * G + w_h * H + w_m * M + w_l * L + w_s * Ssym
    )
    S = 1 / (1 + np.exp(-lam * (Lscore - theta)))
    return S


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
elif selected_example != "None":
    candidates = load_candidates(example_options[selected_example])
else:
    candidates = None

# ---------------------------
# Compute & visualize
# ---------------------------
if candidates is not None:
    # Ensure all required columns exist
    required_cols = ["G", "H", "M", "L", "Ssym"]
    missing = [c for c in required_cols if c not in candidates.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
    else:
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

        st.success(f"Computed site likelihoods for {len(candidates)} candidates.")

        # Show table
        st.dataframe(candidates[[*required_cols, "S"]])

        # Map visualization (if coordinates available)
        if "geometry" in candidates:
            # GeoDataFrame from GeoJSON
            gdf = gpd.GeoDataFrame(candidates)
            gdf = gdf.set_geometry("geometry")
            gdf = gdf.to_crs(epsg=4326)

            fig = px.scatter_mapbox(
                gdf,
                lat=gdf.geometry.y,
                lon=gdf.geometry.x,
                color="S",
                color_continuous_scale="Viridis",
                size="S",
                zoom=6,
                title="Site Likelihood Map",
                mapbox_style="open-street-map",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No geometry column found — skipping map view.")
else:
    st.info("Upload a file or choose an example to begin.")
