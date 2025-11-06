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
# Header and file controls
# ---------------------------
st.title("🌍 Archaeo-Resonance Explorer")

st.markdown(
    "Upload your candidate site data (`.geojson` or `.csv`) or choose a built-in example to compute and visualize site-likelihood scores."
)

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

if "candidates" not in st.session_state:
    st.session_state["candidates"] = None
if "map_mode" not in st.session_state:
    st.session_state["map_mode"] = "Scatter Map"

if uploaded:
    st.session_state["candidates"] = load_candidates(uploaded)
    st.session_state["image_path"] = None
elif selected_example != "None":
    st.session_state["candidates"] = load_candidates(example_options[selected_example])
    st.session_state["image_path"] = os.path.splitext(example_options[selected_example])[0] + ".png"

# ---------------------------
# Tabs for main content
# ---------------------------
tabs = st.tabs(["🗺️ Map View", "📊 Analytics", "ℹ️ About"])

# ---------------------------
# Main app logic
# ---------------------------
candidates = st.session_state["candidates"]

if candidates is not None:
    required_cols = ["G", "H", "M", "L", "Ssym"]
    missing = [c for c in required_cols if c not in candidates.columns]

    if missing:
        st.warning(f"Missing required columns: {', '.join(missing)}")

        if st.button("🧩 Auto-fix missing columns"):
            for col in missing:
                candidates[col] = np.random.uniform(0.3, 0.8, size=len(candidates))
            st.session_state["candidates"] = candidates
            st.success("Added missing columns with random default values.")
            missing = []

    if not missing:
        candidates["S"] = compute_S(
            candidates["G"], candidates["H"], candidates["M"],
            candidates["L"], candidates["Ssym"],
            w_g, w_h, w_m, w_l, w_s, theta, lam,
        )

        st.success(f"✅ Computed site likelihoods for {len(candidates)} candidates.")

        # ------------- MAP VIEW TAB -------------
        with tabs[0]:
            st.subheader("Geospatial Visualization")
            st.session_state["map_mode"] = st.radio(
                "Choose map mode:",
                ["Scatter Map", "Heatmap"],
                horizontal=True,
                index=0 if st.session_state["map_mode"] == "Scatter Map" else 1,
            )

            if "geometry" in candidates:
                try:
                    gdf = gpd.GeoDataFrame(candidates)
                    if gdf.crs is None:
                        gdf.set_crs(epsg=4326, inplace=True)
                    if not all(gdf.geometry.geom_type == "Point"):
                        gdf["geometry"] = gdf.geometry.centroid
                    gdf["lat"] = gdf.geometry.y
                    gdf["lon"] = gdf.geometry.x

                    if st.session_state["map_mode"] == "Scatter Map":
                        fig = px.scatter_mapbox(
                            gdf,
                            lat="lat",
                            lon="lon",
                            color="S",
                            color_continuous_scale="Viridis",
                            size="S",
                            zoom=6,
                            mapbox_style="open-street-map",
                            title="Site Likelihood Scatter Map",
                        )
                    else:
                        fig = px.density_mapbox(
                            gdf,
                            lat="lat",
                            lon="lon",
                            z="S",
                            radius=20,
                            center=dict(lat=gdf["lat"].mean(), lon=gdf["lon"].mean()),
                            zoom=6,
                            mapbox_style="open-street-map",
                            color_continuous_scale="YlOrRd",
                            title="Site Likelihood Heatmap",
                        )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot map: {e}")
            else:
                st.info("No geometry found — showing only table view.")

            # Optional PNG overlay
            if st.session_state.get("image_path") and os.path.exists(st.session_state["image_path"]):
                st.image(Image.open(st.session_state["image_path"]), caption="Associated Site Map", use_container_width=True)

        # ------------- ANALYTICS TAB -------------
        with tabs[1]:
            st.subheader("Statistical Overview")
            st.dataframe(candidates[required_cols + ["S"]])
            st.markdown("#### Distribution of Likelihood Scores")
            st.plotly_chart(px.histogram(candidates, x="S", nbins=20, color_discrete_sequence=["#4B9CD3"]), use_container_width=True)
            st.markdown("#### Component Correlations")
            st.plotly_chart(px.scatter_matrix(candidates, dimensions=required_cols + ["S"], color="S"), use_container_width=True)

            # Export section
            st.markdown("---")
            st.subheader("💾 Export Results")

            csv_buffer = BytesIO()
            candidates.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_buffer.getvalue(),
                file_name="archaeo_resonance_results.csv",
                mime="text/csv",
            )

            try:
                gdf = gpd.GeoDataFrame(candidates)
                gdf.set_crs(epsg=4326, inplace=True)
                geojson_bytes = gdf.to_json().encode("utf-8")
                st.download_button(
                    label="🌐 Download as GeoJSON",
                    data=geojson_bytes,
                    file_name="archaeo_resonance_results.geojson",
                    mime="application/geo+json",
                )
            except Exception as e:
                st.warning(f"Could not export GeoJSON: {e}")

        # ------------- ABOUT TAB -------------
        with tabs[2]:
            st.markdown("""
            ### ℹ️ About Archaeo-Resonance Explorer
            This experimental tool integrates multiple archaeological sensing modalities — geometry, harmonics, magnetic, LIDAR, and symbolic — to compute a unified **site-likelihood score** using a logistic fusion model.

            #### Fusion Formula
            \[
            S = \frac{1}{1 + e^{-\lambda (w_g G + w_h H + w_m M + w_l L + w_s S_{sym} - \theta)}}
            \]

            - **Weights (`w_g` – `w_s`)** represent modality influence.  
            - **θ (bias)** adjusts sensitivity to detection threshold.  
            - **λ (sigmoid)** controls the steepness of the logistic curve.

            ---
            **Developed for exploratory research** in archaeological site prediction and cross-modality resonance modeling.
            """)
else:
    st.info("📂 Upload a file or choose an example to begin.")
